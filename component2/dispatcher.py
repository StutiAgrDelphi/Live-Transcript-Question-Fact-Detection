# component2/dispatcher.py
from abc import ABC, abstractmethod
from typing import List
from shared.schema import Flag
import asyncio
from dataclasses import asdict
from component3.retrieval import is_document_visible

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

    ACCESS CONTROL: this is the ONE place that knows both (a) which document(s)
    a flag's answer/verdict came from, and (b) who is actually on the other end
    of each socket. So per-viewer redaction has to happen here, at send time —
    everywhere upstream (fact-check, QA, doc-lookup) computes ONE shared result
    per flag for the whole meeting, under the resolver identity.
    """

    def __init__(self):
        self._sockets = []  # list of (websocket, user_id, user_role)
        self.questions: List[Flag] = []
        self.facts: List[Flag] = []
        self.document_lookups: List[Flag] = []

    def register(self, websocket, user_id, user_role):
        self._sockets.append((websocket, user_id, user_role))

    def unregister(self, websocket):
        self._sockets = [s for s in self._sockets if s[0] is not websocket]

    def emit(self, flag: Flag):
        buckets = {
            "question": self.questions,
            "fact": self.facts,
            "document_lookup": self.document_lookups,
        }
        buckets.get(flag.type, self.facts).append(flag)
        for ws, user_id, user_role in list(self._sockets):
            asyncio.create_task(self._send_filtered(ws, user_id, user_role, flag))

    async def _send_filtered(self, ws, user_id, user_role, flag: Flag):
        payload = asdict(flag)

        if flag.resolved and flag.sources:
            visible = []
            for s in flag.sources:
                doc = s.get("document")
                ok = await is_document_visible(user_id, user_role, doc) if doc else True
                if ok:
                    visible.append(s)

            if not visible:
                # None of the documents behind this flag are visible to this
                # viewer — withhold the generated text too, not just the link.
                # The verdict/answer text was derived from restricted content,
                # so leaving it in place would leak it via paraphrase even
                # with the citation removed.
                payload["sources"] = []
                payload["reason"] = "Restricted for your access level."
                payload["answer"] = None
                payload["correct_fact"] = None
                payload["citation_document"] = None
                payload["citation_url"] = None
                if payload.get("type") == "document_lookup":
                    payload["document_found"] = False
            else:
                payload["sources"] = visible

        await self._safe_send(ws, payload)

    @staticmethod
    async def _safe_send(ws, payload):
        try:
            await ws.send_json(payload)
        except Exception:
            pass  # socket likely already closed; its own receive loop will unregister it