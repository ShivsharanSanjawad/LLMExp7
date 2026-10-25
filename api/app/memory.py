import threading
import time
from collections import OrderedDict
from typing import Dict, List


class SessionStore:
    """In-memory sliding-window conversation history, one list per session_id. Thread-safe."""

    def __init__(self, max_messages: int, msg_chars: int, ttl_s: int, max_sessions: int):
        self.max_messages = max_messages
        self.msg_chars = msg_chars
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._data: "OrderedDict[str, dict]" = OrderedDict()   # sid -> {"t": last_used, "msgs": [..]}
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        for sid in [s for s, v in self._data.items() if now - v["t"] > self.ttl_s]:
            del self._data[sid]

    def get_history(self, sid: str) -> List[Dict[str, str]]:
        with self._lock:
            now = time.monotonic()
            self._sweep(now)
            entry = self._data.get(sid)
            return [dict(m) for m in entry["msgs"]] if entry else []

    def add_turn(self, sid: str, user_text: str, bot_text: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._sweep(now)
            entry = self._data.setdefault(sid, {"t": now, "msgs": []})
            entry["t"] = now
            entry["msgs"].append({"role": "user", "content": user_text[: self.msg_chars]})
            entry["msgs"].append({"role": "assistant", "content": bot_text[: self.msg_chars]})
            entry["msgs"] = entry["msgs"][-self.max_messages:]
            self._data.move_to_end(sid)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def clear(self, sid: str) -> bool:
        with self._lock:
            return self._data.pop(sid, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
