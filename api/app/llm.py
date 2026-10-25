import logging
import re
from functools import lru_cache

from openai import OpenAI, OpenAIError

from . import config
from .errors import BackendUnavailable

log = logging.getLogger("llm")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                  timeout=config.LLM_TIMEOUT_S, max_retries=0)


def llm_chat(system: str, user: str, max_new_tokens: int = config.GEN_TOKENS) -> str:
    """One chat completion with greedy decoding. Connection/HTTP/timeout errors -> BackendUnavailable."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        resp = get_client().chat.completions.create(
            model=config.LLM_MODEL, messages=messages, temperature=0, max_tokens=max_new_tokens)
    except OpenAIError as e:
        log.warning("LLM call failed: %s", type(e).__name__)
        raise BackendUnavailable("llm") from e
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    return _THINK_RE.sub("", text).strip()


def llm_reachable(timeout: float = 3.0) -> bool:
    try:
        get_client().with_options(timeout=timeout).models.list()
        return True
    except Exception:
        return False
