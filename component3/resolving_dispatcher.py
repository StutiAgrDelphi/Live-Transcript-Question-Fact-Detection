# component3/resolving_dispatcher.py
"""
Drop-in wrapper around any existing Dispatcher. Forwards each flag through
immediately (so the UI shows "detected, verifying..." right away), then
resolves it in the background and emits the SAME flag object again — now
carrying the answer/verdict/link — using flag_id so the UI updates the
existing card instead of creating a duplicate.
"""
import asyncio, logging
from component2.dispatcher import Dispatcher
from shared.schema import Flag
from component3.fact_check_agent import FactCheckAgent
from component3.question_answer_agent import QuestionAnswerAgent
from component3.document_lookup_agent import DocumentLookupAgent

log = logging.getLogger(__name__)

class ResolvingDispatcher(Dispatcher):
    def __init__(self, inner: Dispatcher):
        self.inner = inner
        self.doc_lookup = DocumentLookupAgent()
        self.fact_checker = FactCheckAgent(self.doc_lookup)
        self.answerer = QuestionAnswerAgent(self.doc_lookup)

    def emit(self, flag: Flag):
        self.inner.emit(flag)
        asyncio.create_task(self._resolve(flag))

    def register(self, websocket):
        self.inner.register(websocket)

    def unregister(self, websocket):
        self.inner.unregister(websocket)

    async def _resolve(self, flag: Flag):
        try:
            if flag.type == "fact":
                resolved = await self.fact_checker.check(flag)
            elif flag.type == "question":
                resolved = await self.answerer.answer(flag)
            elif flag.type == "document_lookup":
                resolved = await self.doc_lookup.lookup(flag)
            else:
                return
            self.inner.emit(resolved)  # same flag_id -> UI updates in place
        except Exception as e:
            log.warning(f"Resolution failed for flag {flag.flag_id}: {e}", exc_info=True)
            flag.resolved = True
            flag.reason = "Resolution failed due to an internal error."
            self.inner.emit(flag)  # still re-emit — UI must always leave "checking..."