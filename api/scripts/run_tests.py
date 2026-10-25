"""Endpoint tests that fill the lab's observation tables.
Run from the api/ folder with the API already running:  python -m scripts.run_tests
Add --skip-backend-down to skip the manual 'stop LM Studio' test."""
import argparse
import csv
import os
import sys
import uuid

import httpx
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
TIMEOUT = httpx.Timeout(300.0, connect=10.0)
FORBIDDEN = ["Traceback", "localhost", "127.0.0.1", "api_key", "apikey", "Authorization", "openai"]
if API_KEY:
    FORBIDDEN.append(API_KEY)

RAG_CASES = [
    ("How do I file a claim after a car accident?", "supported"),
    ("I want to cancel my policy. Is there a fee?", "supported"),
    ("Can I pay my premium in monthly installments?", "supported"),
    ("Where can I check the status of my claim?", "supported"),
    ("I cannot log in to my account. What should I do?", "supported"),
    ("My payment seems to have been charged twice. What should I do?", "supported"),
    ("What is the live status of claim 99812?", "out_of_scope"),
    ("Can you tell me the password for my account?", "out_of_scope"),
    ("What's the weather in Mumbai today?", "out_of_scope"),
    ("How can I track my shipment?", "out_of_scope"),
    ("Do you sell home insurance?", "supported (check grounding)"),
    ("insurance", "ambiguous"),
    ("My dog needs coverage", "ambiguous"),
    ("My policy number is AB1234567 and my card is 4111 1111 1111 1111. Why was I charged twice?", "ambiguous"),
]
ANSWERED = ("answered", "answered_after_retry")


def post_ask(question, session_id=None, with_key=True):
    headers = {"X-API-Key": API_KEY} if (API_KEY and with_key) else {}
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    r = httpx.post(f"{API_URL}/ask", json=payload, headers=headers, timeout=TIMEOUT)
    try:
        body = r.json()
    except Exception:
        body = {}
    return r, body


def row(case, expected, actual, passed):
    return {"Test case": case, "Expected result": expected, "Actual result": actual,
            "Pass/Fail": "PASS" if passed else "FAIL"}


def run_lab_tests(skip_backend_down: bool):
    rows = []

    r = httpx.get(f"{API_URL}/health", timeout=10)
    rows.append(row("Health endpoint", "API reports running (200, status ok)",
                    f"HTTP {r.status_code} {r.text[:80]}", r.status_code == 200 and r.json().get("status") == "ok"))

    r, b = post_ask("How do I file a claim after a car accident?", uuid.uuid4().hex)
    ok = r.status_code == 200 and b.get("outcome") in ANSWERED
    rows.append(row("Supported question", "Grounded answer is returned (outcome answered)",
                    f"HTTP {r.status_code} outcome={b.get('outcome')} grounding={b.get('grounding')}", ok))
    latency = b.get("latency_ms")
    rows.append(row("Response-time measurement", "latency_ms is returned and greater than 0",
                    f"latency_ms={latency}", isinstance(latency, int) and latency > 0))

    r, b = post_ask("What's the weather in Mumbai today?", uuid.uuid4().hex)
    rows.append(row("Unsupported question", "Polite refusal / insufficient information (outcome out_of_scope or fallback)",
                    f"HTTP {r.status_code} outcome={b.get('outcome')}",
                    r.status_code == 200 and b.get("outcome") in ("out_of_scope", "fallback")))

    r, b = post_ask("")
    rows.append(row("Empty question", "Validation error (HTTP 422)", f"HTTP {r.status_code}", r.status_code == 422))
    r, b = post_ask("   ")
    rows.append(row("Whitespace-only question", "Validation error (HTTP 422)", f"HTTP {r.status_code}", r.status_code == 422))
    r, b = post_ask("x" * 1001)
    rows.append(row("Question longer than 1000 characters", "Validation error (HTTP 422)",
                    f"HTTP {r.status_code}", r.status_code == 422))

    if API_KEY:
        r, b = post_ask("hello", with_key=False)
        rows.append(row("Missing API key", "HTTP 401", f"HTTP {r.status_code}", r.status_code == 401))

    sid = uuid.uuid4().hex
    post_ask("How do I file a claim after a car accident?", sid)
    r, b = post_ask("how long does it take?", sid)
    rows.append(row("Follow-up question uses memory", "HTTP 200 and rewritten_question is not null",
                    f"HTTP {r.status_code} rewritten_question={b.get('rewritten_question')!r}",
                    r.status_code == 200 and b.get("rewritten_question") is not None))
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    d = httpx.delete(f"{API_URL}/sessions/{sid}", headers=headers, timeout=10)
    rows.append(row("Session reset", "HTTP 200 and cleared is true", f"HTTP {d.status_code} {d.text[:60]}",
                    d.status_code == 200 and d.json().get("cleared") is True))

    if skip_backend_down:
        rows.append({"Test case": "Backend unavailable", "Expected result": "Controlled error; no secret details",
                     "Actual result": "skipped (--skip-backend-down)", "Pass/Fail": "SKIPPED"})
    else:
        input("\nSTOP the LM Studio server now (Developer tab -> Stop Server), then press Enter... ")
        r, b = post_ask("How do I file a claim after a car accident?", uuid.uuid4().hex)
        leaked = [w for w in FORBIDDEN if w.lower() in r.text.lower()]
        rows.append(row("Backend unavailable", "HTTP 503 with a generic message; no secret details",
                        f"HTTP {r.status_code} body={r.text[:100]} leaked={leaked}", r.status_code == 503 and not leaked))
        input("Start the LM Studio server again and load the model, then press Enter... ")
    return rows


def run_rag_cases():
    rows = []
    for q, expected in RAG_CASES:
        r, b = post_ask(q, uuid.uuid4().hex)
        sources = "; ".join(f"row {s['row_id']} ({s['intent']}, {s['score']})" for s in b.get("sources", [])) or "none"
        outcome = b.get("outcome", f"HTTP {r.status_code}")
        rows.append({
            "Customer question": q, "Expected type": expected,
            "Predicted category": b.get("category"), "Predicted intent": b.get("intent"),
            "Retrieved source / relevant passage": sources,
            "Answer supported?": "yes" if outcome in ANSWERED else "no",
            "Retry/fallback?": "retry" if outcome == "answered_after_retry" else ("fallback" if outcome == "fallback" else "no"),
            "Grounding": b.get("grounding"), "Outcome": outcome,
            "Bot answer": (b.get("answer") or "")[:200], "Remarks": "",
        })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backend-down", action="store_true")
    args = ap.parse_args()

    lab = run_lab_tests(args.skip_backend_down)
    for r in lab:
        print(f"[{r['Pass/Fail']}] {r['Test case']}: {r['Actual result']}")
    write_csv("observation_table_lab.csv", lab)

    rag = run_rag_cases()
    for r in rag:
        print(f"{r['Outcome']:<22} | {r['Customer question'][:60]}")
    write_csv("observation_table_rag.csv", rag)

    sys.exit(1 if any(r["Pass/Fail"] == "FAIL" for r in lab) else 0)


if __name__ == "__main__":
    main()
