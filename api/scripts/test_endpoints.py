"""Endpoint checks against a running API. Prints one PASS/FAIL line per check and exits 1 if any fail.

Run from the api/ folder with the API already running:
    uv run python -m scripts.test_endpoints
    uv run python -m scripts.test_endpoints --backend-down   # stop LM Studio first, then run this
"""
import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
TIMEOUT = 120


def headers(extra=None):
    h = {"Content-Type": "application/json", **(extra or {})}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def run_checks(client):
    results = []

    # /health and /ready: 200 when Qdrant and the LLM are up
    for path in ("/health", "/ready"):
        r = client.get(path)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        results.append(check(f"GET {path} reports dependencies up",
                             r.status_code == 200 and body.get("status") == "ok", f"HTTP {r.status_code}"))

    # /ask: grounded answer, returns latency
    r = client.post("/ask", json={"question": "How do I file a claim after a car accident?"}, headers=headers())
    body = r.json() if r.status_code == 200 else {}
    results.append(check("POST /ask grounded question returns 200 with answer",
                         r.status_code == 200 and bool(body.get("answer")), f"HTTP {r.status_code}"))
    results.append(check("POST /ask response includes latency_ms",
                         isinstance(body.get("latency_ms"), int), str(body.get("latency_ms"))))

    # /ask: validation
    r = client.post("/ask", json={"question": "   "}, headers=headers())
    results.append(check("POST /ask blank question returns 422", r.status_code == 422, f"HTTP {r.status_code}"))

    # /ask/stream: step events then one result event, status 200
    with client.stream("POST", "/ask/stream", json={"question": "How do I cancel my policy?"},
                       headers=headers({"Accept": "text/event-stream"})) as r:
        events = [line[len("event:"):].strip() for line in r.iter_lines() if line.startswith("event:")]
        results.append(check("POST /ask/stream returns 200 and a result event",
                             r.status_code == 200 and events and events[-1] == "result",
                             f"HTTP {r.status_code}, last event {events[-1] if events else 'none'}"))

    r = client.post("/ask/stream", json={"question": ""}, headers=headers())
    results.append(check("POST /ask/stream empty question returns 422", r.status_code == 422, f"HTTP {r.status_code}"))

    # /sessions: clear an existing session, then an unknown one
    session_id = "endpoint-test-session"
    client.post("/ask", json={"question": "Hello", "session_id": session_id}, headers=headers())
    r = client.delete(f"/sessions/{session_id}", headers=headers())
    results.append(check("DELETE /sessions/{id} clears an existing session",
                         r.status_code == 200 and r.json().get("cleared") is True, f"HTTP {r.status_code}"))
    r = client.delete("/sessions/no-such-session", headers=headers())
    results.append(check("DELETE /sessions/{id} unknown session returns 404",
                         r.status_code == 404, f"HTTP {r.status_code}"))

    # API key: only meaningful when API_KEY is set
    if API_KEY:
        r = client.post("/ask", json={"question": "hi"})
        results.append(check("POST /ask without API key returns 401", r.status_code == 401, f"HTTP {r.status_code}"))
        r = client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "wrong"})
        results.append(check("POST /ask with wrong API key returns 401", r.status_code == 401, f"HTTP {r.status_code}"))
    else:
        print("SKIP  API key checks (API_KEY is not set in api/.env)")

    return results


def run_backend_down(client):
    """Only meaningful when LM Studio is stopped. Expects controlled 503s with no internal details."""
    r = client.get("/health")
    ok_health = check("GET /health with LLM down returns 503",
                      r.status_code == 503 and r.json().get("llm") == "unreachable", f"HTTP {r.status_code}")
    r = client.post("/ask", json={"question": "How do I file a claim?"}, headers=headers())
    text = json.dumps(r.json()) if r.content else ""
    ok_ask = check("POST /ask with LLM down returns 503 without internal details",
                   r.status_code == 503 and "http" not in text.lower() and "traceback" not in text.lower(),
                   f"HTTP {r.status_code}")
    return [ok_health, ok_ask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-down", action="store_true",
                    help="run the backend-unavailable checks (stop LM Studio first)")
    args = ap.parse_args()

    print(f"Testing {API_URL}\n")
    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
        try:
            results = run_backend_down(client) if args.backend_down else run_checks(client)
        except httpx.ConnectError:
            print(f"Cannot reach {API_URL}. Start the API first.")
            return 1

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
