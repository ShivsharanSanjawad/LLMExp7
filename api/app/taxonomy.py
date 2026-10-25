"""Loads data/taxonomy.json, which scripts/ingest.py writes."""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "taxonomy.json"

if not PATH.exists():
    raise SystemExit(f"{PATH} not found. Run `python -m scripts.ingest` from the api/ folder first.")

_data = json.loads(PATH.read_text(encoding="utf-8"))
CATEGORIES = _data["categories"]
INTENTS = _data["intents"]
INTENTS_BY_CATEGORY = _data["intents_by_category"]
EXAMPLE_Q = _data["example_question"]
