"""Embeddings + Qdrant Cloud access. Everything is created once by init() at server start."""
import logging
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from . import config
from .errors import BackendUnavailable

log = logging.getLogger("retrieval")
_S = {"embeddings": None, "client": None, "store": None}


def init() -> None:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": config.EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    try:
        exists = client.collection_exists(config.COLLECTION)
    except Exception as e:
        raise SystemExit(f"Cannot reach Qdrant ({type(e).__name__}). Check QDRANT_URL and QDRANT_API_KEY.")
    if not exists:
        raise SystemExit(f"Qdrant collection '{config.COLLECTION}' does not exist. "
                         "Run `python -m scripts.ingest` from the api/ folder first.")
    store = QdrantVectorStore(client=client, collection_name=config.COLLECTION, embedding=embeddings)
    embeddings.embed_query("warm up")
    _S.update(embeddings=embeddings, client=client, store=store)
    log.info("retrieval ready: collection=%s", config.COLLECTION)


def search(query: str, k: int, category: Optional[str] = None):
    """Returns a list of (Document, cosine_score). Qdrant failures -> BackendUnavailable."""
    flt = None
    if category:
        flt = Filter(must=[FieldCondition(key="metadata.category", match=MatchValue(value=category))])
    try:
        return _S["store"].similarity_search_with_score(query, k=k, filter=flt)
    except Exception as e:
        log.warning("Qdrant search failed: %s", type(e).__name__)
        raise BackendUnavailable("qdrant") from e


def embed_documents(texts: List[str]):
    return _S["embeddings"].embed_documents(texts)


def qdrant_reachable() -> bool:
    try:
        _S["client"].get_collection(config.COLLECTION)
        return True
    except Exception:
        return False
