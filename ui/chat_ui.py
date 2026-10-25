"""Chainlit chat UI. It contains NO RAG logic: every message is sent to the FastAPI /ask endpoint."""
import os
import uuid

import chainlit as cl
import httpx
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
SHOW_DEBUG = os.environ.get("SHOW_DEBUG", "true").strip().lower() == "true"
TIMEOUT = httpx.Timeout(180.0, connect=10.0)


def _headers() -> dict:
    return {"X-API-Key": API_KEY} if API_KEY else {}


@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", uuid.uuid4().hex)
    await cl.Message(
        content="Hello! I can help with insurance questions about claims, payments, your policy and your account. "
                "For example: \"How do I file a claim after a car accident?\""
    ).send()


@cl.on_chat_end
async def end():
    sid = cl.user_session.get("session_id")
    if not sid:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.delete(f"{API_URL}/sessions/{sid}", headers=_headers())
    except Exception:
        pass   # best effort only


@cl.on_message
async def main(message: cl.Message):
    sid = cl.user_session.get("session_id")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(f"{API_URL}/ask", json={"question": message.content, "session_id": sid},
                                  headers=_headers())
    except httpx.ConnectError:
        await cl.Message(content="I could not reach the assistant service. Please try again later.").send()
        return
    except httpx.TimeoutException:
        await cl.Message(content="The assistant took too long to answer. Please try again.").send()
        return

    if r.status_code == 200:
        data = r.json()
        await cl.Message(content=data["answer"]).send()
        if SHOW_DEBUG:
            lines = [f"outcome: {data.get('outcome')}",
                     f"category: {data.get('category')} | intent: {data.get('intent')}",
                     f"grounding: {data.get('grounding')} | latency: {data.get('latency_ms')} ms",
                     f"steps: {' -> '.join(data.get('path', []))}"]
            if data.get("rewritten_question"):
                lines.append(f"rewritten question: {data['rewritten_question']}")
            for s in data.get("sources", []):
                lines.append(f"source: row {s['row_id']} ({s['intent']}, score {s['score']})")
            async with cl.Step(name="Pipeline details") as step:
                step.output = "\n".join(lines)
    elif r.status_code == 422:
        await cl.Message(content="Please send a message between 1 and 1000 characters.").send()
    elif r.status_code == 401:
        await cl.Message(content="The chat service is not configured correctly (API key rejected).").send()
    elif r.status_code == 503:
        await cl.Message(content="The assistant is temporarily unavailable. Please try again in a moment.").send()
    else:
        await cl.Message(content="Something went wrong. Please try again later.").send()
