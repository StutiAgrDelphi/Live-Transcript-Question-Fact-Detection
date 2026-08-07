# component3/resolving_dispatcher.py
"""
Drop-in wrapper around any existing Dispatcher. Forwards each flag through
immediately (so the UI shows "detected, verifying..." right away), then
resolves it in the background and emits the SAME flag object again — now
carrying the answer/verdict/link — using flag_id so the UI updates the
existing card instead of creating a duplicate.

ACCESS CONTROL: resolution (fact-check / question-answer / doc-lookup) runs
once per flag for the WHOLE meeting, shared across every viewer — there's no
single "current user" at this point. So it runs under a designated RESOLVER
identity (organizer-level access, sees everything), configured via env vars
rather than hardcoded, so this stays swappable per deployment. The actual
per-viewer restriction is enforced afterwards, at send time, in
component2/dispatcher.py — that's the only place that knows who's on the
other end of a specific socket.
"""
import asyncio, logging, os
from component2.dispatcher import Dispatcher
from shared.schema import Flag
from component3.fact_check_agent import FactCheckAgent
from component3.question_answer_agent import QuestionAnswerAgent
from component3.document_lookup_agent import DocumentLookupAgent

log = logging.getLogger(__name__)

# The identity resolution runs under. Must correspond to a role that your RLS
# policy treats as full-access ('organizer' or 'admin') — see the policy on
# ai.general_knowledge. Configurable per environment, not a document/user
# hardcode.
RESOLVER_USER_ID = os.environ.get("RESOLVER_USER_ID", "SYSTEM_RESOLVER")
RESOLVER_ROLE = os.environ.get("RESOLVER_ROLE", "organizer")


class ResolvingDispatcher(Dispatcher):
    def __init__(self, inner: Dispatcher):
        self.inner = inner
        self.doc_lookup = DocumentLookupAgent(RESOLVER_USER_ID, RESOLVER_ROLE)
        self.fact_checker = FactCheckAgent(self.doc_lookup, RESOLVER_USER_ID, RESOLVER_ROLE)
        self.answerer = QuestionAnswerAgent(self.doc_lookup, RESOLVER_USER_ID, RESOLVER_ROLE)

    def emit(self, flag: Flag):
        self.inner.emit(flag)
        asyncio.create_task(self._resolve(flag))

    def register(self, websocket, user_id, user_role):
        self.inner.register(websocket, user_id, user_role)

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