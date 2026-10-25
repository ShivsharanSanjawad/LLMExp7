"""LLM structured output: Pydantic schema + LangChain PydanticOutputParser (same method as Experiment 6)."""
import difflib
import json
import logging
import re
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, create_model

from . import config, taxonomy
from .errors import BackendUnavailable
from .llm import llm_chat

log = logging.getLogger("structured")

SPECIAL = ["smalltalk", "out_of_scope", "unclear"]
CATEGORY_LABELS = taxonomy.CATEGORIES + SPECIAL
INTENTS_BY_CATEGORY = taxonomy.INTENTS_BY_CATEGORY

CategoryChoice = create_model("CategoryChoice", category=(Literal[tuple(CATEGORY_LABELS)], ...))
_intent_models = {}


def intent_model(cat):
    if cat not in _intent_models:
        _intent_models[cat] = create_model(f"IntentChoice_{cat}", intent=(Literal[tuple(INTENTS_BY_CATEGORY[cat])], ...))
    return _intent_models[cat]


class Draft(BaseModel):
    can_answer: bool = Field(description="True only if the passages answer the question.")
    answer: str = Field(description="Answer using only the approved passages, or INSUFFICIENT_CONTEXT.")


def snap_label(value, labels):
    v = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    lower = {l.lower(): l for l in labels}
    if v in lower:
        return lower[v]
    close = difflib.get_close_matches(v, list(lower), n=1, cutoff=0.75)
    return lower[close[0]] if close else None


def scan_label(raw, labels):
    low = raw.lower()
    for l in sorted(labels, key=len, reverse=True):
        if l.lower() in low:
            return l
    return None


def run_label(model_cls, field, labels, system, user, max_new_tokens=40):
    parser = PydanticOutputParser(pydantic_object=model_cls)

    def _run(u):
        raw = llm_chat(system, u, max_new_tokens)
        data = {}
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = {}
        value = snap_label(data.get(field), labels) or scan_label(raw, labels)
        return parser.parse(json.dumps({field: value}))  # validated against the Literal schema

    return RunnableLambda(_run).invoke(user)


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
    'Reply with ONLY a JSON object like {"category": "<category>"}.'
)


def classify_category(q):
    res = with_retries(lambda: run_label(CategoryChoice, "category", CATEGORY_LABELS, CATEGORY_SYSTEM,
                                         f"Customer message: {q}"))
    return res.category if res else None


def classify_intent_in(q, cat):
    ints = INTENTS_BY_CATEGORY[cat]
    if len(ints) == 1:
        return ints[0]
    system = (f"The customer message belongs to the category {cat}. Pick the ONE best intent.\n"
              "Intents with an example question each:\n"
              + "\n".join(f"{i}: e.g. {taxonomy.EXAMPLE_Q.get(i, '')}" for i in ints) + "\n"
              'Reply with ONLY a JSON object like {"intent": "<intent>"}.')
    res = with_retries(lambda: run_label(intent_model(cat), "intent", ints, system, f"Customer message: {q}"))
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
    "- Keep the answer under 80 words.\n"
    'Reply with ONLY a JSON object: {"can_answer": true or false, "answer": "<text>"}.'
)


def run_draft(user):
    parser = PydanticOutputParser(pydantic_object=Draft)

    def _run(u):
        raw = llm_chat(GEN_SYSTEM, u, config.GEN_TOKENS)
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return parser.parse(m.group(0))
            except Exception:
                pass
        m2 = re.search(r'"can_answer"\s*:\s*(true|false).*?"answer"\s*:\s*"(.*?)"\s*\}?\s*$', raw, re.S | re.I)
        if m2:
            return Draft(can_answer=m2.group(1).lower() == "true", answer=m2.group(2))
        raise ValueError("unparseable draft")

    return RunnableLambda(_run).invoke(user)
