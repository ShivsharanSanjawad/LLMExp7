import hashlib
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Generator, List, Optional

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
INTERNAL_MSG = "Internal server error."
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.check_serving_config()
    retrieval.init()
    app.state.graph = build_graph()
    app.state.sessions = SessionStore(config.MAX_HISTORY_MESSAGES, config.HISTORY_MSG_CHARS,
                                      config.SESSION_TTL_S, config.MAX_SESSIONS)
    tracking.init()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    mlflow.langchain.autolog()
    mlflow.openai.autolog()   # covers raw OpenAI client calls (condense_question) that LangChain autolog misses
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


def dependency_status() -> JSONResponse:
    """200 when Qdrant and the LLM are both reachable, otherwise 503 naming what is down."""
    q_ok = retrieval.qdrant_reachable()
    l_ok = llm_reachable()
    body = {"status": "ok" if (q_ok and l_ok) else "unavailable",
            "qdrant": "ok" if q_ok else "unreachable", "llm": "ok" if l_ok else "unreachable"}
    return JSONResponse(status_code=200 if (q_ok and l_ok) else 503, content=body)


class HealthOut(BaseModel):
    status: str = Field(description="'ok' when Qdrant and the LLM are reachable, otherwise 'unavailable'.")
    qdrant: str = Field(description="'ok' or 'unreachable'.")
    llm: str = Field(description="'ok' or 'unreachable'.")


HEALTH_RESPONSES = {503: {"model": HealthOut, "description": "Qdrant or the LLM is unreachable."}}


@app.get("/health", response_model=HealthOut, responses=HEALTH_RESPONSES)
def health():
    """Reports whether the API can answer questions: checks Qdrant and the LLM."""
    return dependency_status()


@app.get("/ready", response_model=HealthOut, responses=HEALTH_RESPONSES)
def ready():
    """Same check as /health, kept for load balancers that call /ready."""
    return dependency_status()


ERROR_RESPONSES = {
    401: {"description": "API key missing or wrong (only when API_KEY is set)."},
    503: {"description": "Backend unavailable. Try again later."},
}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)],
          responses={401: ERROR_RESPONSES[401], 503: ERROR_RESPONSES[503]})
def ask(req: AskRequest, request: Request):
    started = time.perf_counter()
    session_id = req.session_id or uuid.uuid4().hex
    session_tag = hashlib.sha256(session_id.encode()).hexdigest()[:8]
    sessions: SessionStore = request.app.state.sessions

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    history: list = []
    try:
        history = sessions.get_history(session_id)
        state = request.app.state.graph.invoke(
            {"question": req.question, "history": history, "retry_count": 0, "path": []})
        answer = state.get("final_answer") or FALLBACK_MESSAGE
        sessions.add_turn(session_id, mask_cards(req.question), answer)   # history stores masked text only
    except BackendUnavailable as e:
        ms = elapsed_ms()
        log.warning("ask backend_unavailable component=%s latency_ms=%d session=%s", e.component, ms, session_tag)
        tracking.log_request("backend_unavailable", ms, n_history_messages=len(history),
                             session_tag=session_tag, failed_component=e.component)
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MSG, headers={"Retry-After": "10"})
    except Exception:
        ms = elapsed_ms()
        log.exception("ask unexpected_error latency_ms=%d session=%s", ms, session_tag)   # trace stays in server log
        tracking.log_request("error", ms, n_history_messages=len(history), session_tag=session_tag)
        raise HTTPException(status_code=500, detail=INTERNAL_MSG)

    outcome = outcome_of(state)
    latency_ms = elapsed_ms()
    log.info("ask ok outcome=%s category=%s intent=%s latency_ms=%d session=%s",
             outcome, state.get("category"), state.get("intent"), latency_ms, session_tag)
    tracking.log_request("ok", latency_ms, outcome=outcome, category=state.get("category"),
                         intent=state.get("intent"), grounding=state.get("grounding"),
                         retry_count=state.get("retry_count", 0), n_sources=len(state.get("sources", [])),
                         n_history_messages=len(history), session_tag=session_tag)
    return AskResponse(
        answer=answer, outcome=outcome, category=state.get("category"), intent=state.get("intent"),
        grounding=state.get("grounding"), rewritten_question=state.get("rewritten_question"),
        sources=state.get("sources", []), path=state.get("path", []),
        session_id=session_id, latency_ms=int(latency_ms))


@app.post("/ask/stream", dependencies=[Depends(require_api_key)],
          responses={401: ERROR_RESPONSES[401]})
def ask_stream(req: AskRequest, request: Request):
    """SSE endpoint. Streams one 'step' event per pipeline node, then a 'result' event with the full answer.

    The HTTP status is 200 once the stream starts, so a failure after that is sent as an 'error' event
    with code 503 (backend unavailable) or 500 (anything else)."""
    started = time.perf_counter()
    session_id = req.session_id or uuid.uuid4().hex
    session_tag = hashlib.sha256(session_id.encode()).hexdigest()[:8]
    sessions: SessionStore = request.app.state.sessions

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    def event_stream() -> Generator[str, None, None]:
        history: list = []
        initial = {"question": req.question, "history": [], "retry_count": 0, "path": []}
        merged: dict = dict(initial)
        try:
            history = sessions.get_history(session_id)
            initial["history"] = history
            merged = dict(initial)
            for chunk in request.app.state.graph.stream(initial, stream_mode="updates"):
                for name, data in chunk.items():
                    merged.update(data)
                    payload = json.dumps({"node": name, "lines": describe(name, data),
                                          "update": data or {}, "state": merged}, default=str)
                    yield f"event: step\ndata: {payload}\n\n"
            answer = merged.get("final_answer") or FALLBACK_MESSAGE
            sessions.add_turn(session_id, mask_cards(req.question), answer)
        except BackendUnavailable as e:
            ms = elapsed_ms()
            log.warning("stream backend_unavailable component=%s latency_ms=%d session=%s",
                        e.component, ms, session_tag)
            tracking.log_request("backend_unavailable", ms, n_history_messages=len(history),
                                 session_tag=session_tag, failed_component=e.component)
            yield f"event: error\ndata: {json.dumps({'code': 503, 'detail': UNAVAILABLE_MSG})}\n\n"
            return
        except Exception:
            ms = elapsed_ms()
            log.exception("stream unexpected_error latency_ms=%d session=%s", ms, session_tag)
            tracking.log_request("error", ms, n_history_messages=len(history), session_tag=session_tag)
            yield f"event: error\ndata: {json.dumps({'code': 500, 'detail': INTERNAL_MSG})}\n\n"
            return

        outcome = outcome_of(merged)
        latency_ms = int(elapsed_ms())
        log.info("stream ok outcome=%s category=%s intent=%s latency_ms=%d session=%s",
                 outcome, merged.get("category"), merged.get("intent"), latency_ms, session_tag)
        tracking.log_request("ok", latency_ms, outcome=outcome, category=merged.get("category"),
                             intent=merged.get("intent"), grounding=merged.get("grounding"),
                             retry_count=merged.get("retry_count", 0), n_sources=len(merged.get("sources", [])),
                             n_history_messages=len(history), session_tag=session_tag)
        result = AskResponse(
            answer=answer, outcome=outcome, category=merged.get("category"), intent=merged.get("intent"),
            grounding=merged.get("grounding"), rewritten_question=merged.get("rewritten_question"),
            sources=merged.get("sources", []), path=merged.get("path", []),
            session_id=session_id, latency_ms=latency_ms)
        yield f"event: result\ndata: {result.model_dump_json()}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)],
            responses={401: ERROR_RESPONSES[401], 404: {"description": "No session with that id."}})
def clear_session(request: Request,
                  session_id: str = Path(..., min_length=1, max_length=64, pattern=SESSION_ID_PATTERN)):
    if not request.app.state.sessions.clear(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"cleared": True}
