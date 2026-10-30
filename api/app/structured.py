"""LLM structured output via ChatOpenAI.with_structured_output(method="json_schema").
LM Studio enforces the JSON schema at the grammar/token level — output is always valid.
"""
import logging
from functools import lru_cache
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, APITimeoutError
from pydantic import BaseModel, Field, create_model

from . import config, taxonomy
from .errors import BackendUnavailable

log = logging.getLogger("structured")

SPECIAL = ["smalltalk", "out_of_scope", "unclear"]
CATEGORY_LABELS = taxonomy.CATEGORIES + SPECIAL
INTENTS_BY_CATEGORY = taxonomy.INTENTS_BY_CATEGORY


@lru_cache(maxsize=2)
def _llm(max_tokens: int) -> ChatOpenAI:
    """Cached ChatOpenAI instance keyed by max_tokens."""
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=0,
        max_tokens=max_tokens,
        max_retries=0,
        timeout=config.LLM_TIMEOUT_S,
    )


CategoryChoice = create_model("CategoryChoice", category=(Literal[tuple(CATEGORY_LABELS)], ...))
_intent_models: dict = {}


def intent_model(cat):
    if cat not in _intent_models:
        _intent_models[cat] = create_model(
            f"IntentChoice_{cat}", intent=(Literal[tuple(INTENTS_BY_CATEGORY[cat])], ...))
    return _intent_models[cat]


class Draft(BaseModel):
    can_answer: bool = Field(description="True only if the passages answer the question.")
    answer: str = Field(description="Answer using only the approved passages, or INSUFFICIENT_CONTEXT.")


def _invoke(schema, system: str, user: str, max_tokens: int):
    """Call LLM with json_schema structured output. Raises BackendUnavailable on connection/server errors."""
    try:
        structured = _llm(max_tokens).with_structured_output(schema, method=config.LLM_STRUCTURED_METHOD)
        return structured.invoke([SystemMessage(system), HumanMessage(user)])
    except (APIConnectionError, APITimeoutError) as e:
        raise BackendUnavailable("llm") from e
    except APIError as e:
        if e.status_code in (401, 403, 429, 502, 503, 504):   # bad key, rate limit, provider down
            log.warning("LLM provider returned HTTP %s", e.status_code)
            raise BackendUnavailable("llm") from e
        raise  # 400 etc. → propagate to with_retries


def with_retries(fn):
    """Retry parse failures up to MAX_TRIES. Backend (connection) errors are NOT swallowed."""
    for _ in range(config.MAX_TRIES):
        try:
            return fn()
        except BackendUnavailable:
            raise
        except Exception as e:
            log.debug("structured output attempt failed: %s", type(e).__name__)
    return None


CATEGORY_SYSTEM = (
    "You route messages sent to an insurance customer-support bot.\n"
    "Pick the ONE best category for the customer message.\n"
    "Categories and the intents inside them:\n"
    + "\n".join(f"{c}: {', '.join(INTENTS_BY_CATEGORY[c])}" for c in taxonomy.CATEGORIES) + "\n"
    "smalltalk: greetings, thanks, chit-chat.\n"
    "out_of_scope: anything unrelated to insurance support (weather, sports, cooking, coding, general knowledge).\n"
    "unclear: insurance-related but too vague to tell what the customer wants (for example the single word 'insurance').\n"
)


def classify_category(q):
    res = with_retries(lambda: _invoke(CategoryChoice, CATEGORY_SYSTEM, f"Customer message: {q}", max_tokens=40))
    return res.category if res else None


def classify_intent_in(q, cat):
    ints = INTENTS_BY_CATEGORY[cat]
    if len(ints) == 1:
        return ints[0]
    system = (
        f"The customer message belongs to the category {cat}. Pick the ONE best intent.\n"
        "Intents with an example question each:\n"
        + "\n".join(f"{i}: e.g. {taxonomy.EXAMPLE_Q.get(i, '')}" for i in ints) + "\n"
    )
    res = with_retries(lambda: _invoke(intent_model(cat), system, f"Customer message: {q}", max_tokens=40))
    return res.intent if res else None


def classify(q):
    """Returns (category, intent). Both come from LLM structured output."""
    cat = classify_category(q)
    if cat is None:
        return None, None
    if cat in SPECIAL:
        return cat, cat
    return cat, classify_intent_in(q, cat)


GEN_SYSTEM = (
    "You are a polite insurance customer-support assistant.\n"
    "Answer the customer question using ONLY the approved passages. Reuse their wording.\n"
    "Rules:\n"
    "- Do not invent policy terms, amounts, dates, timelines, links, phone numbers or account status.\n"
    "- You cannot see customer accounts or live records.\n"
    "- Never ask for passwords, one-time codes or card details.\n"
    "- If the passages do not answer the question, set can_answer to false and answer to INSUFFICIENT_CONTEXT.\n"
    "- Write the answer in Markdown: short paragraphs, a numbered list for step-by-step procedures, "
    "and bullet points for related items.\n"
    "- When the passages list several options, items or attributes, present them as a Markdown table with a header row.\n"
    "- Be complete: cover every relevant point the passages state. Do not pad.\n"
    "- Only state facts the passages state. If they do not say whether a fee, charge, amount, deadline or condition "
    "applies, say that the passages do not specify it and direct the customer to the support team.\n"
    "- Never say that something has no fee, is free, or is not charged unless a passage says so.\n"
    "- Answer every part of the customer question. If the question asks something the passages do not cover "
    "(for example whether a fee applies), add one sentence saying the help content does not cover it and the "
    "customer should contact the support team for that.\n"
)


def run_draft(user):
    """Single invocation; caller wraps in with_retries()."""
    return _invoke(Draft, GEN_SYSTEM, user, max_tokens=config.GEN_TOKENS)


class CoveragePart(BaseModel):
    part: str = Field(description="One fact or sub-question the customer asks for, in a few words.")
    covered: bool = Field(description="True only if the passages explicitly state this fact.")


class Coverage(BaseModel):
    parts: List[CoveragePart] = Field(description="At most 4 parts. Empty if the question asks for no specific fact.")


COVERAGE_SYSTEM = (
    "You check whether help-content passages answer a customer question.\n"
    "Split the question into the specific facts it asks for. Write each part as a short noun phrase, "
    "for example 'the early termination fee amount' or 'the claim approval timeline'. "
    "Mark each part covered only if the passages explicitly state that fact. "
    "Passages that only mention a topic in general do not cover a specific fact.\n"
)


def run_coverage(user):
    """Single invocation; caller wraps in with_retries()."""
    return _invoke(Coverage, COVERAGE_SYSTEM, user, max_tokens=config.GEN_TOKENS)
