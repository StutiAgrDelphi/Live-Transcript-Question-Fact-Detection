# component3/fact_check_agent.py
import os, json
from shared.schema import Flag
from component3.retrieval import search_knowledge_base
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient
from component3.document_lookup_agent import DocumentLookupAgent

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

Return ONLY JSON: {"verdict": "correct"|"incorrect"|"unverifiable", "correct_fact": "..."|null}
"""

class FactCheckAgent:
    def __init__(self, doc_lookup: "DocumentLookupAgent"):
        self.doc_lookup = doc_lookup
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )
        self.agent = create_harness_agent(client=client, agent_instructions=SYSTEM_PROMPT, name="FactChecker")

    async def check(self, flag: Flag) -> Flag:
        document_name = await self.doc_lookup.resolve(flag.resolved_text)
        chunks = await search_knowledge_base(flag.resolved_text, top_k=5, document_name=document_name)
        if not chunks:
            flag.verdict = "unverifiable"
            flag.resolved = True
            return flag

        context = "\n\n".join(f"[{c.document_name}] {c.content}" for c in chunks)
        prompt = f"CLAIM: {flag.resolved_text}\n\nRETRIEVED PASSAGES:\n{context}"
        response = await self.agent.run(prompt, session=self.agent.create_session())
        data = json.loads(response.text.strip().strip("`").removeprefix("json").strip())

        best = chunks[0]
        flag.verdict = data["verdict"]
        flag.correct_fact = data.get("correct_fact")
        flag.citation_document = best.document_name
        flag.citation_url = best.document_url
        flag.resolved = True
        return flag