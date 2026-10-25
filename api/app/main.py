import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import mlflow
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from . import config, retrieval, tracking
from .errors import BackendUnavailable
from .graph import build_graph
from .guards import FALLBACK_MESSAGE, mask_cards
from .llm import llm_reachable
from .memory import SessionStore
from .nodes import outcome_of
from .trace import describe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

UNAVAILABLE_MSG = "The assistant is temporarily unavailable. Please try again later."
BUSY_MSG = "The assistant is busy. Please try again shortly."
INTERNAL_MSG = "Internal server error."
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.check_serving_config()
    retrieval.init()
    app.state.graph = build_graph()
    app.state.sessions = SessionStore(config.MAX_HISTORY_MESSAGES, config.HISTORY_MSG_CHARS,
                                      config.SESSION_TTL_S, config.MAX_SESSIONS)
    # Semaphore allows up to LLM_PARALLEL concurrent graph runs (matches LM Studio's Parallel setting).
    # Falls back to 1 if LLM_PARALLEL <= 0.
    parallel = max(1, config.LLM_PARALLEL)
    app.state.graph_lock = threading.Semaphore(parallel)
    log.info("concurrency: up to %d parallel graph runs", parallel)
    tracking.init()
    # Enable MLflow auto-tracing for LangChain/LangGraph — captures all node executions as spans.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    mlflow.langchain.autolog()
    log.info("MLflow LangGraph autolog enabled")
    log.info("startup complete")
    yield


app = FastAPI(title="Insurance Support Bot API", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e.get("loc", ()) if p != "body"),
               "message": str(e.get("msg", "invalid value"))} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = config.API_KEY
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=64, pattern=SESSION_ID_PATTERN)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class SourceOut(BaseModel):
    row_id: int
    intent: str
    score: float
    passage: str


class AskResponse(BaseModel):
    answer: str
    outcome: str
    category: Optional[str] = None
    intent: Optional[str] = None
    grounding: Optional[float] = None
    rewritten_question: Optional[str] = None
    sources: List[SourceOut] = []
    path: List[str] = []
    session_id: str
    latency_ms: int


@app.get("/health")
def health():
    """Liveness only: the process is up. It does NOT check the LLM or Qdrant (see /ready)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    q_ok = retrieval.qdrant_reachable()
    l_ok = llm_reachable()
    body = {"status": "ready" if (q_ok and l_ok) else "not_ready",
            "qdrant": "ok" if q_ok else "unreachable", "llm": "ok" if l_ok else "unreachable"}
    return JSONResponse(status_code=200 if (q_ok and l_ok) else 503, content=body)


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest, request: Request):
    started = time.perf_counter()
    session_id = req.session_id or uuid.uuid4().hex
    session_tag = hashlib.sha256(session_id.encode()).hexdigest()[:8]
    sessions: SessionStore = request.app.state.sessions
    sem: threading.Semaphore = request.app.state.graph_lock

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    if not sem.acquire(timeout=config.QUEUE_TIMEOUT_S):
        ms = elapsed_ms()
        log.warning("ask busy latency_ms=%d session=%s", ms, session_tag)
        tracking.log_request("busy", ms, session_tag=session_tag)
        raise HTTPException(status_code=503, detail=BUSY_MSG, headers={"Retry-After": "5"})

    queue_ms = elapsed_ms()
    history: list = []
    graph_started = time.perf_counter()
    try:
        history = sessions.get_history(session_id)
        state = request.app.state.graph.invoke(
            {"question": req.question, "history": history, "retry_count": 0, "path": []})
        answer = state.get("final_answer") or FALLBACK_MESSAGE
        sessions.add_turn(session_id, mask_cards(req.question), answer)   # history stores masked text only
    except BackendUnavailable as e:
        ms = elapsed_ms()
        log.warning("ask backend_unavailable component=%s latency_ms=%d session=%s", e.component, ms, session_tag)
        tracking.log_request("backend_unavailable", ms, queue_ms=queue_ms, n_history_messages=len(history),
                             session_tag=session_tag, failed_component=e.component)
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MSG, headers={"Retry-After": "10"})
    except Exception:
        ms = elapsed_ms()
        log.exception("ask unexpected_error latency_ms=%d session=%s", ms, session_tag)   # trace stays in server log
        tracking.log_request("error", ms, queue_ms=queue_ms, n_history_messages=len(history), session_tag=session_tag)
        raise HTTPException(status_code=500, detail=INTERNAL_MSG)
    finally:
        sem.release()

    graph_ms = (time.perf_counter() - graph_started) * 1000
    outcome = outcome_of(state)
    latency_ms = elapsed_ms()

    log.info("ask ok outcome=%s category=%s intent=%s latency_ms=%d graph_ms=%d session=%s",
             outcome, state.get("category"), state.get("intent"), latency_ms, graph_ms, session_tag)
    tracking.log_request("ok", latency_ms, graph_ms=graph_ms, queue_ms=queue_ms, outcome=outcome,
                         category=state.get("category"), intent=state.get("intent"),
                         grounding=state.get("grounding"), retry_count=state.get("retry_count", 0),
                         n_sources=len(state.get("sources", [])), n_history_messages=len(history),
                         session_tag=session_tag)
    return AskResponse(
        answer=answer, outcome=outcome, category=state.get("category"), intent=state.get("intent"),
        grounding=state.get("grounding"), rewritten_question=state.get("rewritten_question"),
        sources=state.get("sources", []), path=state.get("path", []),
        session_id=session_id, latency_ms=int(latency_ms))


@app.post("/ask/stream", dependencies=[Depends(require_api_key)])
async def ask_stream(req: AskRequest, request: Request):
    """SSE endpoint. Streams pipeline step events then a final 'result' event with the full response JSON."""
    started = time.perf_counter()
    session_id = req.session_id or uuid.uuid4().hex
    session_tag = hashlib.sha256(session_id.encode()).hexdigest()[:8]
    sessions: SessionStore = request.app.state.sessions
    sem: threading.Semaphore = request.app.state.graph_lock

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    async def event_stream() -> AsyncGenerator[str, None]:
        if not sem.acquire(timeout=config.QUEUE_TIMEOUT_S):
            yield f"event: error\ndata: {json.dumps({'code': 503, 'detail': BUSY_MSG})}\n\n"
            return

        queue_ms = elapsed_ms()
        history: list = []
        graph_started = time.perf_counter()
        state = None
        try:
            history = sessions.get_history(session_id)
            initial = {"question": req.question, "history": history, "retry_count": 0, "path": []}

            # Stream node-by-node using LangGraph's sync .stream() in the thread
            import concurrent.futures
            import asyncio

            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def run_graph():
                try:
                    for chunk in request.app.state.graph.stream(initial, stream_mode="updates"):
                        for node_name in chunk:
                            loop.call_soon_threadsafe(queue.put_nowait, ("step", node_name, chunk[node_name]))
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None, None))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc), None))

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            fut = loop.run_in_executor(executor, run_graph)

            # Accumulate state from updates
            merged: dict = dict(initial)
            while True:
                kind, name, data = await queue.get()
                if kind == "step":
                    merged.update(data)
                    payload = json.dumps({"node": name, "lines": describe(name, data),
                                          "update": data or {}, "state": merged}, default=str)
                    yield f"event: step\ndata: {payload}\n\n"
                elif kind == "done":
                    break
                else:  # error
                    yield f"event: error\ndata: {json.dumps({'code': 500, 'detail': name})}\n\n"
                    return

            await fut  # ensure executor is done
            state = merged
            answer = state.get("final_answer") or FALLBACK_MESSAGE
            sessions.add_turn(session_id, mask_cards(req.question), answer)

        except BackendUnavailable as e:
            ms = elapsed_ms()
            sem.release()
            log.warning("stream backend_unavailable component=%s latency_ms=%d session=%s",
                        e.component, ms, session_tag)
            tracking.log_request("backend_unavailable", ms, queue_ms=queue_ms,
                                  n_history_messages=len(history), session_tag=session_tag,
                                  failed_component=e.component)
            yield f"event: error\ndata: {json.dumps({'code': 503, 'detail': UNAVAILABLE_MSG})}\n\n"
            return
        except Exception:
            ms = elapsed_ms()
            sem.release()
            log.exception("stream unexpected_error latency_ms=%d session=%s", ms, session_tag)
            tracking.log_request("error", ms, queue_ms=queue_ms,
                                  n_history_messages=len(history), session_tag=session_tag)
            yield f"event: error\ndata: {json.dumps({'code': 500, 'detail': INTERNAL_MSG})}\n\n"
            return
        finally:
            sem.release()

        graph_ms = (time.perf_counter() - graph_started) * 1000
        outcome = outcome_of(state)
        latency_ms = int(elapsed_ms())

        log.info("stream ok outcome=%s category=%s intent=%s latency_ms=%d session=%s",
                 outcome, state.get("category"), state.get("intent"), latency_ms, session_tag)
        tracking.log_request("ok", latency_ms, graph_ms=graph_ms, queue_ms=queue_ms, outcome=outcome,
                             category=state.get("category"), intent=state.get("intent"),
                             grounding=state.get("grounding"), retry_count=state.get("retry_count", 0),
                             n_sources=len(state.get("sources", [])), n_history_messages=len(history),
                             session_tag=session_tag)

        result = AskResponse(
            answer=answer, outcome=outcome, category=state.get("category"), intent=state.get("intent"),
            grounding=state.get("grounding"), rewritten_question=state.get("rewritten_question"),
            sources=state.get("sources", []), path=state.get("path", []),
            session_id=session_id, latency_ms=latency_ms)
        yield f"event: result\ndata: {result.model_dump_json()}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
def clear_session(request: Request,
                  session_id: str = Path(..., min_length=1, max_length=64, pattern=SESSION_ID_PATTERN)):
    return {"cleared": request.app.state.sessions.clear(session_id)}
