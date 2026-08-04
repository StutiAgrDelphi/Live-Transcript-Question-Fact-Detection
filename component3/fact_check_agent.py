# component3/fact_check_agent.py
import os, json
from shared.schema import Flag
from component3.retrieval import search_knowledge_base
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient

SYSTEM_PROMPT = """You are fact-checking one claim from a live meeting against retrieved
document excerpts.

- If the passages support the claim as stated -> verdict "correct".
- If they contradict it (wrong figure, date, name, etc.) -> verdict "incorrect", and give
  the correct fact per the documents.
- If the passages don't address the claim -> verdict "unverifiable". Never guess.

Return ONLY JSON: {"verdict": "correct"|"incorrect"|"unverifiable", "correct_fact": "..."|null}
"""

class FactCheckAgent:
    def __init__(self):
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )
        self.agent = create_harness_agent(client=client, agent_instructions=SYSTEM_PROMPT, name="FactChecker")

    async def check(self, flag: Flag) -> Flag:
        chunks = await search_knowledge_base(flag.resolved_text, top_k=5)
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