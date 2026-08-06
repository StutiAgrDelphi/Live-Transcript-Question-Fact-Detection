# component3/question_answer_agent.py
import os, json
from shared.schema import Flag
from component3.retrieval import search_knowledge_base
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient
from agent_framework import ChatOptions
from component3.document_lookup_agent import DocumentLookupAgent, _extract_json

SYSTEM_PROMPT = """Answer the question using ONLY the retrieved passages below.
If the passages directly answer the question, answer confidently and directly — don't
hedge if the evidence is there. Only express uncertainty if the passages genuinely do
not address the question.

Return ONLY JSON: {"answer": "...", "answerable": true|false}
"""

class QuestionAnswerAgent:
    def __init__(self, doc_lookup: "DocumentLookupAgent"):
        self.doc_lookup = doc_lookup
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )
        self.agent = create_harness_agent(client=client, agent_instructions=SYSTEM_PROMPT, name="QuestionAnswerer")

    async def answer(self, flag: Flag) -> Flag:
        candidates = await self.doc_lookup.resolve_candidates(flag.resolved_text)
        if not candidates:
            flag.answer = "Could not identify which document this question refers to."
            flag.resolved = True
            return flag

        chunks = await search_knowledge_base(flag.resolved_text, top_k=5, document_names=candidates)
        if not chunks:
            flag.answer = "No relevant information found in the knowledge base."
            flag.reason = f"Identified {', '.join(candidates)}, but found no passages addressing this question."
            flag.resolved = True
            return flag

        context = "\n\n".join(f"[{c.document_name}] {c.content}" for c in chunks)
        prompt = f"QUESTION: {flag.resolved_text}\n\nRETRIEVED PASSAGES:\n{context}"
        response = await self.agent.run(prompt, session=self.agent.create_session(), options=ChatOptions(temperature=0))
        data = _extract_json(response.text) or {"answer": "Could not parse a response."}

        seen = {}
        for c in chunks:
            seen.setdefault(c.document_name, c.document_url)

        flag.answer = data.get("answer", "Could not parse a response.")
        flag.sources = [{"document": d, "url": u} for d, u in seen.items()]
        flag.resolved = True
        return flag