"""Human-readable trace of each LangGraph node, streamed to the UI thinking panel."""
from . import config


def describe(node: str, update: dict) -> list:
    """Detail lines for one node's state update (the dict the node returned)."""
    update = update or {}
    if node == "condense_question":
        if update.get("rewritten_question"):
            return [f"Follow-up rewritten as: \"{update['rewritten_question']}\""]
        return ["Question is already standalone (no rewrite)"]
    if node == "classify_intent":
        if update.get("final_answer"):
            return [f"Handled without retrieval (intent: {update.get('intent')})"]
        if update.get("intent") == "unclassified":
            return ["Classification failed after retries"]
        return [f"Category: {update.get('category') or update.get('intent')} · intent: {update.get('intent')}"]
    if node == "clarify":
        return ["Question is too vague, asking the customer a follow-up"]
    if node == "retrieve":
        srcs = update.get("sources", [])
        if not srcs:
            return [f"No passage reached the score threshold ({config.SCORE_THRESHOLD})"]
        lines = [f"Kept {len(srcs)} passage(s) from Qdrant"]
        lines += [f"row {s['row_id']} · intent {s['intent']} · cosine {s['score']}" for s in srcs]
        return lines
    if node == "draft_answer":
        return ["Draft answer generated from the passages" if update.get("can_answer")
                else "Passages judged insufficient for an answer"]
    if node == "review_answer":
        lines = [f"Grounding {update.get('grounding')} (min {config.GROUNDING_MIN}) → {update.get('review')}"]
        lines += [f"Unsupported claim: \"{c}\"" for c in update.get("unsupported_claims", [])]
        return lines
    if node == "check_coverage":
        missing = update.get("uncovered_parts", [])
        if not missing:
            return ["Every fact in the question is covered by the passages"]
        return [f"Not covered by the passages: {', '.join(missing)}"]
    if node == "rewrite_query":
        return [f"Retrying retrieval with: \"{update.get('search_query')}\""]
    if node == "fallback":
        return ["No supported answer, using the fallback message"]
    return []
