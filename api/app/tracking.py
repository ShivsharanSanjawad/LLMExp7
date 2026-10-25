"""MLflow tracking. One run per /ask request. Failures here must never break a request."""
import logging
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from . import config

log = logging.getLogger("tracking")
_S = {"client": None, "experiment_id": None}


def init() -> None:
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        exp = mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        client = MlflowClient()
        run = client.create_run(exp.experiment_id, tags={"mlflow.runName": "server_start"})
        params = {
            "llm_model": config.LLM_MODEL, "llm_base_url_host": config.LLM_BASE_URL.split("//")[-1].split("/")[0],
            "embed_model": config.EMBED_MODEL, "collection": config.COLLECTION,
            "score_threshold": config.SCORE_THRESHOLD, "grounding_min": config.GROUNDING_MIN,
            "retrieve_k": config.RETRIEVE_K, "keep": config.KEEP, "gen_tokens": config.GEN_TOKENS,
            "max_history_messages": config.MAX_HISTORY_MESSAGES, "condense_max_words": config.CONDENSE_MAX_WORDS,
        }
        for k, v in params.items():
            client.log_param(run.info.run_id, k, v)
        client.set_terminated(run.info.run_id)
        _S.update(client=client, experiment_id=exp.experiment_id)
        log.info("MLflow tracking enabled: %s", config.MLFLOW_TRACKING_URI)
    except Exception as e:
        _S["client"] = None
        log.warning("MLflow disabled (%s)", type(e).__name__)


def log_request(status: str, latency_ms: float, graph_ms: Optional[float] = None, queue_ms: Optional[float] = None,
                outcome: Optional[str] = None, category: Optional[str] = None, intent: Optional[str] = None,
                grounding: Optional[float] = None, retry_count: int = 0, n_sources: int = 0,
                n_history_messages: int = 0, session_tag: str = "", failed_component: Optional[str] = None) -> None:
    """status is one of: ok, backend_unavailable, busy, error."""
    client = _S["client"]
    if client is None:
        return
    try:
        run = client.create_run(_S["experiment_id"], tags={"mlflow.runName": "ask"})
        rid = run.info.run_id
        tags = {"status": status, "outcome": outcome or "none", "category": category or "none",
                "intent": intent or "none", "session": session_tag, "failed_component": failed_component or "none"}
        for k, v in tags.items():
            client.set_tag(rid, k, v)
        metrics: Dict[str, float] = {"latency_ms": latency_ms, "retry_count": retry_count,
                                     "n_sources": n_sources, "n_history_messages": n_history_messages,
                                     "success": 1.0 if status == "ok" else 0.0}
        if graph_ms is not None:
            metrics["graph_ms"] = graph_ms
        if queue_ms is not None:
            metrics["queue_ms"] = queue_ms
        if grounding is not None:
            metrics["grounding"] = grounding
        for k, v in metrics.items():
            client.log_metric(rid, k, float(v))
        client.set_terminated(rid, status="FINISHED" if status == "ok" else "FAILED")
    except Exception as e:
        log.warning("MLflow logging failed (%s)", type(e).__name__)
