"""Chainlit chat UI. It contains NO RAG logic: every message is sent to the FastAPI /ask/stream SSE endpoint."""
import asyncio
import json
import os
import re
import uuid

import chainlit as cl
import httpx
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
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
    data = None

    # ── 1. Thinking step: opens immediately and updates live as each node fires ──
    async with cl.Step(name="Thinking...", type="llm", show_input=False, default_open=True) as step:
        live_lines: list = []

        async def push_line(line: str):
            live_lines.append(line)
            step.output = "\n".join(live_lines)
            await step.update()

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{API_URL}/ask/stream",
                    json={"question": message.content, "session_id": sid},
                    headers={**_headers(), "Accept": "text/event-stream"},
                ) as r:
                    if r.status_code != 200:
                        step.name = f"Error {r.status_code}"
                        step.output = f"The server returned {r.status_code}."
                        await cl.Message(content="Something went wrong. Please try again.").send()
                        return

                    event_type = None
                    async for raw_line in r.aiter_lines():
                        raw_line = raw_line.strip()
                        if raw_line.startswith("event:"):
                            event_type = raw_line[len("event:"):].strip()
                        elif raw_line.startswith("data:") and event_type:
                            payload = json.loads(raw_line[len("data:"):].strip())

                            if event_type == "step":
                                node = payload.get("node", "")
                                # One collapsible dropdown per LangGraph node: its update and the full state after it
                                async with cl.Step(name=node, type="tool",
                                                   parent_id=step.id, show_input=False,
                                                   default_open=False) as child:
                                    child.output = (
                                        "**Summary**\n\n"
                                        + "\n".join(f"- {d}" for d in payload.get("lines", []))
                                        + "\n\n**Node update**\n```json\n"
                                        + json.dumps(payload.get("update", {}), indent=2, ensure_ascii=False)
                                        + "\n```\n\n**Full state after this node**\n```json\n"
                                        + json.dumps(payload.get("state", {}), indent=2, ensure_ascii=False)
                                        + "\n```"
                                    )

                            elif event_type == "result":
                                data = payload

                            elif event_type == "error":
                                code = payload.get("code", 500)
                                detail = payload.get("detail", "Unknown error")
                                step.name = "Error"
                                step.output = detail
                                err_msg = {
                                    503: "The assistant is temporarily unavailable. Please try again.",
                                    401: "The chat service is not configured correctly (API key rejected).",
                                }.get(code, "Something went wrong. Please try again.")
                                await cl.Message(content=err_msg).send()
                                return

                            event_type = None

        except httpx.ConnectError:
            step.name = "Connection error"
            step.output = "Could not reach the assistant service. Check the API is running on port 8000."
            await cl.Message(content="I could not reach the assistant service. Please try again later.").send()
            return
        except httpx.TimeoutException:
            step.name = "Request timed out"
            step.output = "The server took too long to respond."
            await cl.Message(content="The assistant took too long to answer. Please try again.").send()
            return

        if data is None:
            step.name = "No response"
            step.output = "No result was received from the server."
            await cl.Message(content="Something went wrong. Please try again.").send()
            return

        # Finalise the step: only the reasoning lines stay, shown like a model's "Thought for Ns" block
        latency_s = round(data.get("latency_ms", 0) / 1000, 2)
        step.name = f"Thought for {latency_s}s"
        if data.get("path"):
            await push_line("Full pipeline: " + " → ".join(data["path"]))
        async with cl.Step(name="Final response (API result)", type="tool", parent_id=step.id,
                           show_input=False, default_open=False) as final_child:
            final_child.output = "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"

    # ── 2. Stream the final answer token-by-token after the thinking step closes ──
    answer = data.get("answer", "")
    msg = cl.Message(content="")
    await msg.send()

    for token in re.findall(r"\S+|\s+", answer):
        await msg.stream_token(token)
        await asyncio.sleep(0.015)

    await msg.update()
