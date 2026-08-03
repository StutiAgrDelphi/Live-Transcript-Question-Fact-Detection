# component2/dispatcher.py
from abc import ABC, abstractmethod
from typing import List
from shared.schema import Flag
import asyncio
from dataclasses import asdict

_TAB_NAMES = {
    "question": "QUESTIONS",
    "fact": "FACTS",
    "document_lookup": "DOCUMENT LOOKUP",
}


class Dispatcher(ABC):
    @abstractmethod
    def emit(self, flag: Flag):
        pass


class ConsoleDispatcher(Dispatcher):
    def __init__(self):
        self.questions: List[Flag] = []
        self.facts: List[Flag] = []
        self.document_lookups: List[Flag] = []

    def emit(self, flag: Flag):
        buckets = {
            "question": self.questions,
            "fact": self.facts,
            "document_lookup": self.document_lookups,
        }
        bucket = buckets.get(flag.type, self.facts)
        bucket.append(flag)
        tab = _TAB_NAMES.get(flag.type, "FACTS")
        print(f"[{tab}] ({flag.speaker} @ {flag.elapsed_seconds:.1f}s) {flag.resolved_text}")


# component2/dispatcher.py  — append below ConsoleDispatcher

class WebSocketDispatcher(Dispatcher):
    """
    Broadcasts each detected Flag to every WebSocket consumer currently subscribed
    to this session (multiple consumers — e.g. a debug UI and the real product UI —
    can watch the same live session at once).
    """

    def __init__(self):
        self._sockets = []
        self.questions: List[Flag] = []
        self.facts: List[Flag] = []
        self.document_lookups: List[Flag] = []

    def register(self, websocket):
        self._sockets.append(websocket)

    def unregister(self, websocket):
        if websocket in self._sockets:
            self._sockets.remove(websocket)

    def emit(self, flag: Flag):
        buckets = {
            "question": self.questions,
            "fact": self.facts,
            "document_lookup": self.document_lookups,
        }
        buckets.get(flag.type, self.facts).append(flag)
        payload = asdict(flag)
        for ws in list(self._sockets):
            asyncio.create_task(self._safe_send(ws, payload))

    @staticmethod
    async def _safe_send(ws, payload):
        try:
            await ws.send_json(payload)
        except Exception:
            pass  # socket likely already closed; its own receive loop will unregister it
