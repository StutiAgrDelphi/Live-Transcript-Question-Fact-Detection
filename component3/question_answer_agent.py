# component3/question_answer_agent.py
import os, json
from shared.schema import Flag
from component3.retrieval import search_knowledge_base
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient
from component3.document_lookup_agent import DocumentLookupAgent

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
        document_name = await self.doc_lookup.resolve(flag.resolved_text)
        chunks = await search_knowledge_base(flag.resolved_text, top_k=5, document_name=document_name)
        if not chunks:
            flag.answer = "No relevant information found in the knowledge base."
            flag.resolved = True
            return flag

        context = "\n\n".join(f"[{c.document_name}] {c.content}" for c in chunks)
        prompt = f"QUESTION: {flag.resolved_text}\n\nRETRIEVED PASSAGES:\n{context}"
        response = await self.agent.run(prompt, session=self.agent.create_session())
        data = json.loads(response.text.strip().strip("`").removeprefix("json").strip())

        best = chunks[0]
        flag.answer = data["answer"]
        flag.citation_document = best.document_name
        flag.citation_url = best.document_url
        flag.resolved = True
        return flag