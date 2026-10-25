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


def sentence_split(text: str, min_words: int = 3):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= min_words]
