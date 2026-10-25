"""LangGraph nodes. Same logic as Experiment 6, plus the new condense_question node."""
import re
from typing import List, Optional, TypedDict

import numpy as np
from langgraph.graph import END

from . import config, retrieval, taxonomy
from .guards import (CARD_RE, CREDENTIAL_REFUSAL, CRED_REQUEST_RE, FALLBACK_MESSAGE, LIVE_CLAIM_RE,
                     LIVE_REPLY, LIVE_RE, OUT_OF_SCOPE_REPLY, RESET_RE, SMALLTALK_REPLY)
from .llm import llm_chat
from .structured import classify, run_coverage, run_draft, with_retries
from .utils import dedupe_by_intent, humanize, sentence_split, strip_markdown, ungrounded_claims


class SupportState(TypedDict, total=False):
    question: str              # raw question from the user
    history: List[dict]        # [{"role": "user"|"assistant", "content": str}], oldest first
    standalone_question: str   # masked question, rewritten if it was a follow-up
    rewritten_question: str    # set only when condense_question actually changed the question
    intent: str
    category: Optional[str]
    search_query: str
    context: List[str]
    sources: List[dict]
    draft: str
    can_answer: bool
    grounding: float
    review: str
    unsupported_claims: List[str]
    uncovered_parts: List[str]
    retry_count: int
    final_answer: str
    clarified: bool
    path: List[str]


CONDENSE_SYSTEM = (
    "You rewrite a customer's latest message into ONE standalone question for an insurance support bot.\n"
    "You are given the recent conversation and the latest customer message.\n"
    "Rules:\n"
    "- Use the conversation only to fill in what the latest message leaves out "
    "(for example what 'it', 'that one' or 'the first one' refers to).\n"
    "- If the latest message is already a complete question on its own, output it unchanged.\n"
    "- Do not answer the question. Do not add information that is not in the conversation.\n"
    "- Output ONLY the rewritten question as a single line of plain text."
)
CONDENSE_TOKENS = 80


def clean_rewrite(raw: str) -> Optional[str]:
    """Validate the model's rewrite. Returns None if it looks wrong (caller then keeps the original)."""
    line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
    line = re.sub(r"^(standalone question|rewritten question|question)\s*:\s*", "", line, flags=re.I)
    line = line.strip().strip('"').strip("'").strip()
    if not line or len(line) > 300 or len(line.split()) > 40:
        return None
    if re.search(r"\b(customer|assistant)\s*:", line, re.I) or "INSUFFICIENT_CONTEXT" in line:
        return None
    return CARD_RE.sub("[CARD NUMBER]", line)


def condense_question(state: SupportState):
    path = state.get("path", []) + ["condense_question"]
    q = state["question"]
    masked = CARD_RE.sub("[CARD NUMBER]", q)
    base = {"path": path, "standalone_question": masked}
    history = state.get("history") or []
    if not history:
        return base
    if CRED_REQUEST_RE.search(q) or LIVE_RE.search(q):      # safety guards decide these later
        return base
    if len(masked.split()) > config.CONDENSE_MAX_WORDS:      # long message = already standalone
        return base
    convo = "\n".join(f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history)
    user = f"Recent conversation:\n{convo}\n\nLatest customer message: {masked}\n\nStandalone question:"
    rewritten = clean_rewrite(llm_chat(CONDENSE_SYSTEM, user, CONDENSE_TOKENS))
    if rewritten is None or rewritten.lower() == masked.lower():
        return base
    return {**base, "standalone_question": rewritten, "rewritten_question": rewritten}


def classify_intent(state: SupportState):
    path = state.get("path", []) + ["classify_intent"]
    q = state["question"]

    if CRED_REQUEST_RE.search(q) and not RESET_RE.search(q):
        return {"path": path, "intent": "sensitive", "category": None, "final_answer": CREDENTIAL_REFUSAL}
    if LIVE_RE.search(q):
        return {"path": path, "intent": "live_status", "category": None, "final_answer": LIVE_REPLY}

    masked = CARD_RE.sub("[CARD NUMBER]", state.get("standalone_question") or q)
    category, intent = classify(masked)
    base = {"path": path, "search_query": masked}

    if category is None:                       # structured output failed twice
        return {**base, "intent": "unclassified", "category": None}
    if category == "smalltalk":
        return {**base, "intent": "smalltalk", "category": None, "final_answer": SMALLTALK_REPLY}
    if category == "out_of_scope":
        return {**base, "intent": "out_of_scope", "category": None, "final_answer": OUT_OF_SCOPE_REPLY}
    if category == "unclear":
        return {**base, "intent": "unclear", "category": None}
    return {**base, "intent": intent or "unclassified", "category": category}


def route_after_classify(state: SupportState):
    if state.get("final_answer"):
        return END
    if state.get("intent") == "unclear":
        return "clarify"
    return "retrieve"


def _source(d, s):
    return {"row_id": d.metadata["row_id"], "intent": d.metadata["intent"], "score": round(float(s), 3),
            "passage": d.metadata["response"][:160]}


def clarify(state: SupportState):
    path = state.get("path", []) + ["clarify"]
    hits = retrieval.search(state.get("search_query") or state["question"], config.RETRIEVE_K)
    kept = dedupe_by_intent(hits, config.KEEP) if hits and hits[0][1] >= config.SCORE_THRESHOLD else []
    topics = [humanize(d.metadata["intent"]) for d, _ in kept]
    if topics:
        listed = ", ".join(topics[:-1]) + (" or " if len(topics) > 1 else "") + topics[-1]
        msg = f"I want to point you to the right help. Are you asking about {listed}? A few more details would help."
    else:
        msg = ("Could you tell me a bit more about what you need? For example, are you asking about a claim, "
               "a payment, your policy or your account?")
    return {"path": path, "clarified": True, "final_answer": msg, "sources": [_source(d, s) for d, s in kept]}


def retrieve(state: SupportState):
    path = state.get("path", []) + ["retrieve"]
    query = state.get("search_query") or state["question"]
    category = state.get("category") if state.get("retry_count", 0) == 0 else None   # retry is unfiltered

    hits = retrieval.search(query, config.RETRIEVE_K, category)
    top = hits[0][1] if hits else 0.0
    kept = dedupe_by_intent(hits, config.KEEP) if top >= config.SCORE_THRESHOLD else []
    return {"path": path,
            "context": [d.metadata["response"] for d, _ in kept],
            "sources": [_source(d, s) for d, s in kept]}


def draft_answer(state: SupportState):
    path = state.get("path", []) + ["draft_answer"]
    if not state.get("context"):
        return {"path": path, "draft": "INSUFFICIENT_CONTEXT", "can_answer": False}
    passages = "\n".join(f"- {c}" for c in state["context"])
    user = f"Customer question: {state.get('search_query', state['question'])}\n\nApproved passages:\n{passages}"
    res = with_retries(lambda: run_draft(user))
    if res is None:
        return {"path": path, "draft": "INSUFFICIENT_CONTEXT", "can_answer": False}
    return {"path": path, "draft": res.answer.strip(), "can_answer": res.can_answer}


def review_answer(state: SupportState):
    path = state.get("path", []) + ["review_answer"]
    draft = state.get("draft", "")
    grounding, review = 0.0, "insufficient"

    if state.get("can_answer") and draft and "INSUFFICIENT_CONTEXT" not in draft and not LIVE_CLAIM_RE.search(draft):
        sents = sentence_split(strip_markdown(draft))
        pass_sents = []
        for c in state.get("context", []):
            pass_sents.extend(sentence_split(c, min_words=2) or [c])
        if sents and pass_sents:
            sv = np.array(retrieval.embed_documents(sents))
            pv = np.array(retrieval.embed_documents(pass_sents))
            grounding = float((sv @ pv.T).max(axis=1).min())   # weakest sentence decides
            review = "supported" if grounding >= config.GROUNDING_MIN else "insufficient"

    unsupported = ungrounded_claims(strip_markdown(draft), state.get("context", [])) if review == "supported" else []
    if unsupported:
        review = "insufficient"   # a number or fee claim that no passage states is never shown

    out = {"path": path, "review": review, "grounding": round(grounding, 3), "unsupported_claims": unsupported}
    if review == "supported":
        out["final_answer"] = draft.strip()
    return out


def check_coverage(state: SupportState):
    """Model checks each fact the customer asked for against the passages; uncovered facts get a plain note."""
    path = state.get("path", []) + ["check_coverage"]
    question = state.get("search_query") or state["question"]
    passages = "\n".join(f"- {c}" for c in state.get("context", []))
    user = f"Customer question: {question}\n\nApproved passages:\n{passages}"
    res = with_retries(lambda: run_coverage(user))
    parts = res.parts if res else []
    missing = [p.part for p in parts if not p.covered]
    if not missing:
        return {"path": path, "uncovered_parts": []}
    if len(missing) == len(parts):   # none of the facts asked for are in the passages: the topic is wrong
        return {"path": path, "uncovered_parts": missing, "review": "insufficient",
                "final_answer": FALLBACK_MESSAGE}
    note = ("The help content does not cover: " + "; ".join(missing)
            + ". Please contact our support team for these details.")
    return {"path": path, "uncovered_parts": missing, "final_answer": f"{state['final_answer']}\n\n{note}"}


def route_after_review(state: SupportState):
    if state.get("review") == "supported":
        return "check_coverage"
    return "rewrite_query" if state.get("retry_count", 0) < config.MAX_RETRIES else "fallback"


def rewrite_query(state: SupportState):
    path = state.get("path", []) + ["rewrite_query"]
    base = state.get("search_query") or state["question"]
    extra = []
    if state.get("category"):
        extra.append(humanize(state["category"]))
    if state.get("intent") in taxonomy.INTENTS:
        extra.append(humanize(state["intent"]))
    return {"path": path, "search_query": " ".join([base] + extra).strip(),
            "retry_count": state.get("retry_count", 0) + 1}


def fallback(state: SupportState):
    return {"path": state.get("path", []) + ["fallback"], "final_answer": FALLBACK_MESSAGE}


def outcome_of(state) -> str:
    intent = state.get("intent")
    if intent == "sensitive":
        return "refused_sensitive"
    if intent == "live_status":
        return "live_status_reply"
    if intent == "smalltalk":
        return "smalltalk"
    if intent == "out_of_scope":
        return "out_of_scope"
    if state.get("clarified"):
        return "clarify"
    if state.get("review") == "supported":
        return "answered_after_retry" if state.get("retry_count", 0) > 0 else "answered"
    return "fallback"
