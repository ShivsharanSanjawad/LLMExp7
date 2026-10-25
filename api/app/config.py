"""All configuration comes from environment variables (loaded from api/.env)."""
import os

from dotenv import load_dotenv

load_dotenv()


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


# ---- LLM backend: LM Studio local server (OpenAI-compatible) ----
LLM_BASE_URL = _str("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = _str("LLM_MODEL")
LLM_API_KEY = _str("LLM_API_KEY", "lm-studio")
LLM_TIMEOUT_S = _float("LLM_TIMEOUT_S", 120)
GEN_TOKENS = _int("GEN_TOKENS", 300)

# ---- Qdrant Cloud ----
QDRANT_URL = _str("QDRANT_URL")
QDRANT_API_KEY = _str("QDRANT_API_KEY")
COLLECTION = _str("COLLECTION", "insurance_support_bot")

# ---- Embeddings and dataset ----
EMBED_MODEL = _str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = _str("EMBED_DEVICE", "cpu")  # cpu keeps GPU memory free for LM Studio
DATASET = "bitext/Bitext-insurance-llm-chatbot-training-dataset"

# ---- RAG pipeline parameters (same meaning as in Experiment 6) ----
RETRIEVE_K = _int("RETRIEVE_K", 6)
KEEP = _int("KEEP", 3)
SCORE_THRESHOLD = _float("SCORE_THRESHOLD", 0.45)
GROUNDING_MIN = _float("GROUNDING_MIN", 0.60)
MAX_TRIES = _int("MAX_TRIES", 2)      # structured-output attempts per LLM call
MAX_RETRIES = _int("MAX_RETRIES", 1)  # rewrite_query runs at most once

# ---- Conversation memory (sliding window, no summarization) ----
MAX_HISTORY_MESSAGES = _int("MAX_HISTORY_MESSAGES", 6)  # 6 messages = 3 question/answer pairs
HISTORY_MSG_CHARS = _int("HISTORY_MSG_CHARS", 300)
CONDENSE_MAX_WORDS = _int("CONDENSE_MAX_WORDS", 15)
SESSION_TTL_S = _int("SESSION_TTL_S", 3600)
MAX_SESSIONS = _int("MAX_SESSIONS", 500)

# ---- API ----
API_KEY = _str("API_KEY")  # empty = API key check disabled
QUEUE_TIMEOUT_S = _float("QUEUE_TIMEOUT_S", 120)

# ---- Parallel LLM requests (matches LM Studio's "Parallel" setting) ----
LLM_PARALLEL = _int("LLM_PARALLEL", 4)

# ---- MLflow ----
MLFLOW_TRACKING_URI = _str("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = _str("MLFLOW_EXPERIMENT", "insurance-support-api")


def _problems(need_llm: bool) -> list:
    problems = []
    if need_llm and (not LLM_MODEL or LLM_MODEL == "REPLACE_ME"):
        problems.append("LLM_MODEL is not set (copy the model identifier from LM Studio)")
    if not QDRANT_URL:
        problems.append("QDRANT_URL is not set")
    if not QDRANT_API_KEY:
        problems.append("QDRANT_API_KEY is not set")
    return problems


def check_serving_config() -> None:
    problems = _problems(need_llm=True)
    if problems:
        raise SystemExit("Configuration error:\n- " + "\n- ".join(problems))


def check_ingest_config() -> None:
    problems = _problems(need_llm=False)
    if problems:
        raise SystemExit("Configuration error:\n- " + "\n- ".join(problems))
