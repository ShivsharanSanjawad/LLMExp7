import re


def humanize(label: str) -> str:
    return label.replace("_", " ").lower()


def dedupe_by_intent(hits, keep):
    seen, out = set(), []
    for doc, score in hits:
        if doc.metadata["intent"] in seen:
            continue
        seen.add(doc.metadata["intent"])
        out.append((doc, score))
        if len(out) == keep:
            break
    return out


def strip_markdown(text: str) -> str:
    """Plain text for grounding: drop table rules, bullets, headings and emphasis so they do not count as sentences."""
    text = re.sub(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", "", text, flags=re.M)   # table separator rows
    text = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", text, flags=re.M)     # list markers
    return re.sub(r"[|#*_`>]+", " ", text)


def sentence_split(text: str, min_words: int = 3):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= min_words]


NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
NEG_CLAIM_RE = re.compile(
    r"\b(?:no|not|without|zero|free|waived)\b(?:\s+\w+){0,2}?\s+(?:fees?|charges?|costs?|penalt(?:y|ies)|payments?)\b"
    r"|\bfree of charge\b|\bnot charged\b|\bwaived\b", re.I)


def ungrounded_claims(draft: str, passages) -> list:
    """Numbers and no-fee style negations that the draft states but no passage contains.
    Embedding similarity cannot catch these, so they are checked as plain text."""
    text = " ".join(passages)
    out = [n for n in NUM_RE.findall(draft) if n not in text]
    for m in NEG_CLAIM_RE.finditer(draft):
        if not NEG_CLAIM_RE.search(text):
            out.append(m.group(0))
    return out


