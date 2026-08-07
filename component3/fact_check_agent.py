# component3/fact_check_agent.py
import os, json
from shared.schema import Flag
from component3.retrieval import search_knowledge_base
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient
from component3.document_lookup_agent import DocumentLookupAgent, _extract_json
from agent_framework import ChatOptions

SYSTEM_PROMPT = """You are fact-checking one claim from a live meeting against retrieved
document excerpts.

Rules:
- "incorrect" means the passages DIRECTLY address the same specific thing the claim is
  about, and state something that contradicts it.
- "unverifiable" means the passages do NOT directly address what the claim is about —
  even if they share vocabulary or are loosely on-topic. Absence of confirmation is NOT
  the same as contradiction. Meeting logistics, opinions, and plans ("we should spend
  more time on X", "let's regroup tomorrow") are almost always unverifiable, not
  incorrect, unless a passage specifically discusses that same decision.
- "correct" means the passages directly support the claim as stated.

When genuinely unsure whether passages address the claim, prefer "unverifiable" over
"incorrect" — calling something wrong when it was simply never addressed is worse than
saying you can't confirm it.

Return ONLY JSON: {"verdict": "correct"|"incorrect"|"unverifiable", "correct_fact": "..."|null, "reason": "one short sentence explaining the verdict"}"""

class FactCheckAgent:
    def __init__(self, doc_lookup: "DocumentLookupAgent", resolver_user_id: str, resolver_role: str):
        self.doc_lookup = doc_lookup
        self.resolver_user_id = resolver_user_id
        self.resolver_role = resolver_role
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )
        self.agent = create_harness_agent(client=client, agent_instructions=SYSTEM_PROMPT, name="FactChecker")

    async def check(self, flag: Flag) -> Flag:
        candidates = await self.doc_lookup.resolve_candidates(flag.resolved_text)
        if not candidates:
            flag.verdict = "unverifiable"
            flag.reason = "Could not identify which document this claim refers to."
            flag.resolved = True
            return flag

        # Resolution runs under the RESOLVER identity (organizer-level) since
        # this result is computed once and shared across every viewer in the
        # meeting. Per-viewer redaction happens later, at delivery time.
        chunks = await search_knowledge_base(
            self.resolver_user_id,
            self.resolver_role,
            flag.resolved_text,
            top_k=5,
            document_names=candidates,
        )
        if not chunks:
            flag.verdict = "unverifiable"
            flag.reason = f"Identified {', '.join(candidates)}, but found no passages addressing this specific point."
            flag.sources = [{"document": c, "url": None} for c in candidates]
            flag.resolved = True
            return flag

        context = "\n\n".join(f"[{c.document_name}] {c.content}" for c in chunks)
        prompt = f"CLAIM: {flag.resolved_text}\n\nRETRIEVED PASSAGES:\n{context}"
        response = await self.agent.run(prompt, session=self.agent.create_session(), options=ChatOptions(temperature=0))
        data = _extract_json(response.text) or {"verdict": "unverifiable"}

        seen = {}
        for c in chunks:
            seen.setdefault(c.document_name, c.document_url)

        flag.verdict = data.get("verdict", "unverifiable")
        flag.correct_fact = data.get("correct_fact")
        flag.reason = data.get("reason")
        flag.sources = [{"document": d, "url": u} for d, u in seen.items()]
        flag.resolved = True
        return flag