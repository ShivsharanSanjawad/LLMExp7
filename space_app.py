"""Gradio front-end for Hugging Face Spaces (ZeroGPU / Gradio SDK).

Replaces the Chainlit UI + FastAPI server on the Space: the LangGraph pipeline from api/app is called
in-process. The local setup (api/ + ui/) is unchanged and still works.
"""
import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))

import gradio as gr

try:
    import spaces
except ImportError:          # running locally without the Spaces package
    spaces = None

from app import config, retrieval                      # noqa: E402  (api/app, see sys.path above)
from app.errors import BackendUnavailable              # noqa: E402
from app.graph import build_graph                      # noqa: E402
from app.guards import FALLBACK_MESSAGE, mask_cards    # noqa: E402
from app.memory import SessionStore                    # noqa: E402
from app.trace import describe                         # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("space")

UNAVAILABLE_MSG = "The assistant is temporarily unavailable. Please try again later."
INTERNAL_MSG = "Something went wrong. Please try again."
SHOW_DEBUG = os.environ.get("SHOW_DEBUG", "true").lower() == "true"

import threading

GRAPH = None
SESSIONS = None
INIT_ERROR = None


def get_pipeline():
    global GRAPH, SESSIONS, INIT_ERROR
    if GRAPH is not None and SESSIONS is not None:
        return GRAPH, SESSIONS
    if INIT_ERROR is not None:
        raise RuntimeError(INIT_ERROR)
    try:
        log.info("Checking configuration and initializing retrieval...")
        config.check_serving_config()
        retrieval.init()
        GRAPH = build_graph()
        SESSIONS = SessionStore(config.MAX_HISTORY_MESSAGES, config.HISTORY_MSG_CHARS,
                                config.SESSION_TTL_S, config.MAX_SESSIONS)
        log.info("LangGraph pipeline successfully loaded.")
        return GRAPH, SESSIONS
    except (Exception, SystemExit) as e:
        INIT_ERROR = str(e)
        log.error("Initialization error: %s", e)
        print(f"CRITICAL INITIALIZATION ERROR: {e}", flush=True)
        raise RuntimeError(str(e)) from e

# ZeroGPU only accepts a Space that has at least one @spaces.GPU function. The pipeline itself needs no GPU
# (LLM = hosted API, embeddings = CPU), so this is a never-called placeholder.
if spaces is not None:
    @spaces.GPU
    def _gpu_placeholder():
        return True


def respond(message, history, request: gr.Request):
    message = (message or "").strip()
    if not message:
        yield "Please type a question."
        return
    message = message[:1000]
    sid = getattr(request, "session_hash", None) or uuid.uuid4().hex

    try:
        graph, sessions = get_pipeline()
    except (Exception, SystemExit) as e:
        log.warning("get_pipeline error: %s", e)
        yield f"Configuration or startup issue: {e}. Please check your environment variables in Render."
        return

    initial = {"question": message, "history": sessions.get_history(sid), "retry_count": 0, "path": []}
    merged = dict(initial)
    trace_lines = []
    try:
        for chunk in graph.stream(initial, stream_mode="updates"):
            for name, data in chunk.items():
                merged.update(data or {})
                for line in describe(name, data):
                    trace_lines.append(f"- **{name}**: {line}")
                yield "*Thinking...*\n\n" + "\n".join(trace_lines)
    except BackendUnavailable as e:
        log.warning("backend_unavailable component=%s", e.component)
        yield UNAVAILABLE_MSG
        return
    except Exception:
        log.exception("unexpected error")
        yield INTERNAL_MSG
        return

    answer = merged.get("final_answer") or FALLBACK_MESSAGE
    sessions.add_turn(sid, mask_cards(message), answer)   # history stores masked text only
    if SHOW_DEBUG and merged.get("path"):
        answer += "\n\n---\n*Pipeline: " + " -> ".join(merged["path"]) + "*"
    yield answer


demo = gr.ChatInterface(
    respond,
    title="Insurance Support Bot",
    description="Ask about claims, payments, your policy and your account. "
                "Answers come only from a curated help knowledge base.",
    examples=["How do I file a claim after a car accident?"],
    cache_examples=False,
)
demo.queue(default_concurrency_limit=4)

def _warmup():
    try:
        get_pipeline()
    except (Exception, SystemExit) as e:
        log.error("Background warmup failed: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    threading.Thread(target=_warmup, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=port)

