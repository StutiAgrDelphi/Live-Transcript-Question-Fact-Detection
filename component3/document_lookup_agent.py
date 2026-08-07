# component3/document_lookup_agent.py
"""
Not semantic search — an entity match. Returns however many documents (0, 1, or
more) a spoken line plausibly refers to. Zero or multiple candidates are both
legitimate, common outcomes — the caller decides what to do with each case
rather than this agent forcing a single answer.

ACCESS CONTROL: entity matching runs under the RESOLVER identity (organizer-
level, sees the full document index) because a single meeting has many
viewers with different permissions and there's no one "current viewer" at
resolution time. Matching a document by NAME is not a content leak — the
actual redaction of restricted content happens later, per-viewer, at
delivery time in component2/dispatcher.py.
"""
import json
import os
import re
from typing import Dict, List, Optional

from agent_framework import create_harness_agent, ChatOptions
from agent_framework_openai import OpenAIChatClient

from shared.schema import Flag
from component3.retrieval import list_known_documents

SYSTEM_PROMPT = """You're given a line from a meeting and a list of document names available
in the knowledge base.

Return every document from the list that the line clearly and specifically refers to — by
company name, policy name, or title keywords. A generic reference like "the annual report" or
"that policy" with nothing identifying which one does NOT count as a match to anything, even
if only one candidate seems likely — return an empty list rather than guessing.

If the line plausibly refers to two or more documents (e.g. it's comparing them, or the
reference is genuinely ambiguous between them), return all of them — do not arbitrarily pick one.

Return ONLY JSON: {"matched_document_names": ["..."]}   (empty list if nothing clearly matches)
"""


class DocumentLookupAgent:
    def __init__(self, resolver_user_id: str, resolver_role: str):
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )
        self.agent = create_harness_agent(client=client, agent_instructions=SYSTEM_PROMPT, name="DocumentMatcher")
        self.resolver_user_id = resolver_user_id
        self.resolver_role = resolver_role
        self._index: Dict[str, str] = {}

    async def refresh(self):
        docs = await list_known_documents(self.resolver_user_id, self.resolver_role)
        self._index = {d["document_name"]: d["document_url"] for d in docs if d["document_name"]}

    async def resolve_candidates(self, text: str) -> List[str]:
        """Returns 0+ known document_names this text plausibly refers to.
        Used by FactCheckAgent and QuestionAnswerAgent to scope retrieval,
        and by lookup() below for the document_lookup flag type itself."""
        if not self._index:
            await self.refresh()

        names = "\n".join(f"- {n}" for n in self._index.keys())
        prompt = f"LINE: {text}\n\nKNOWN DOCUMENTS:\n{names}"
        response = await self.agent.run(
            prompt, session=self.agent.create_session(), options=ChatOptions(temperature=0)
        )
        data = _extract_json(response.text)
        matched = data.get("matched_document_names", []) if data else []
        return [m for m in matched if m in self._index]

    async def lookup(self, flag: Flag) -> Flag:
        candidates = await self.resolve_candidates(flag.resolved_text)
        flag.document_found = len(candidates) == 1  # ambiguous (2+) is not a "found" result either

        if flag.document_found:
            name = candidates[0]
            flag.citation_document = name
            flag.citation_url = self._index[name]
            flag.sources = [{"document": name, "url": self._index[name]}]
        elif len(candidates) > 1:
            flag.reason = f"Ambiguous — could match any of: {', '.join(candidates)}"
            flag.sources = [{"document": c, "url": self._index.get(c)} for c in candidates]
        else:
            flag.reason = "No document in the knowledge base matches this reference."

        flag.resolved = True
        return flag


def _extract_json(text: str) -> Optional[dict]:
    """Robust-ish JSON extraction — pulls the first {...} block instead of
    assuming the LLM wrapped output exactly as asked (fences, stray text, etc.
    vary run to run). Returns None on failure rather than raising, so callers
    can handle it as 'no result' instead of hanging."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None