"""One-off script: build the knowledge base in Qdrant Cloud and write data/taxonomy.json.
Run from the api/ folder:  python -m scripts.ingest
It DROPS and rebuilds the Qdrant collection every time it runs."""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from app import config

SEED = 42
KB_PER_INTENT = 60
CHUNK_SIZE, CHUNK_OVERLAP = 450, 60
PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "taxonomy.json"


def fill_placeholders(text: str) -> str:
    return PLACEHOLDER_RE.sub(lambda m: "our website" if "URL" in m.group(1) else "our support team", text)


def main() -> None:
    config.check_ingest_config()

    raw = load_dataset(config.DATASET, split="train").to_pandas()
    assert {"instruction", "intent", "category", "response"} <= set(raw.columns), list(raw.columns)
    raw = raw.reset_index(names="row_id")

    intents = sorted(raw["intent"].unique())
    categories = sorted(raw["category"].unique())
    intents_by_category = raw.groupby("category")["intent"].agg(lambda s: sorted(s.unique())).to_dict()
    print("rows:", len(raw), "| intents:", len(intents), "| categories:", len(categories))

    kb_parts = []
    for intent, g in raw.groupby("intent"):
        g = g.sample(frac=1, random_state=SEED)
        kb_parts.append(g.iloc[:KB_PER_INTENT])
    kb_df = pd.concat(kb_parts).reset_index(drop=True)
    kb_df["response"] = kb_df["response"].map(fill_placeholders)
    print("knowledge base rows:", len(kb_df))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs, chunk_counts = [], []
    for r in kb_df.itertuples(index=False):
        n = len(splitter.split_text(r.response))
        chunk_counts.append(n)
        docs.append(Document(page_content=r.instruction, metadata={
            "row_id": int(r.row_id), "intent": r.intent, "category": r.category,
            "response": r.response, "answer_chunks": n}))
    print("documents:", len(docs), "| answer chunks per row (min/median/max):",
          min(chunk_counts), int(np.median(chunk_counts)), max(chunk_counts))

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": config.EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(config.COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    client.create_payload_index(config.COLLECTION, field_name="metadata.category",
                                field_schema=PayloadSchemaType.KEYWORD)
    store = QdrantVectorStore(client=client, collection_name=config.COLLECTION, embedding=embeddings)
    t0 = time.time()
    store.add_documents(docs)
    count = client.count(config.COLLECTION).count
    print(f"indexed {count} points in {time.time() - t0:.1f}s")
    assert count == len(docs), "point count does not match document count"

    taxonomy = {
        "categories": categories,
        "intents": intents,
        "intents_by_category": intents_by_category,
        "example_question": kb_df.groupby("intent")["instruction"].first().to_dict(),
    }
    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TAXONOMY_PATH.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print("wrote", TAXONOMY_PATH)


if __name__ == "__main__":
    main()
