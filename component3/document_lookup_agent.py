# component3/document_lookup_agent.py
"""
Not semantic search — an entity match. The question isn't "what content is
related to this," it's "did the speaker name a document that exists in the
KB, and what's its link." An LLM pick against the (short) list of known
document names is more reliable here than embedding similarity on a whole
spoken sentence.
"""
import json, os
from typing import Dict
from shared.schema import Flag
from component3.retrieval import list_known_documents
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient

SYSTEM_PROMPT = """You're given a line from a meeting where someone referenced a document,
and a list of document names actually available in the knowledge base.

Pick the single document from the list that the speaker meant, if any clearly match.
If nothing in the list plausibly matches, say so — do not force a match.

Return ONLY JSON: {"matched_document_name": "..."|null}
"""

class DocumentLookupAgent:
    def __init__(self):
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )
        self.agent = create_harness_agent(client=client, agent_instructions=SYSTEM_PROMPT, name="DocumentMatcher")
        self._index: Dict[str, str] = {}

    async def refresh(self):
        docs = await list_known_documents()
        self._index = {d["document_name"]: d["document_url"] for d in docs if d["document_name"]}

    async def lookup(self, flag: Flag) -> Flag:
        if not self._index:
            await self.refresh()

        names = "\n".join(f"- {n}" for n in self._index.keys())
        prompt = f"LINE: {flag.resolved_text}\n\nKNOWN DOCUMENTS:\n{names}"
        response = await self.agent.run(prompt, session=self.agent.create_session())
        data = json.loads(response.text.strip().strip("`").removeprefix("json").strip())

        matched = data.get("matched_document_name")
        flag.document_found = matched is not None and matched in self._index
        if flag.document_found:
            flag.citation_document = matched
            flag.citation_url = self._index[matched]
        flag.resolved = True
        return flag