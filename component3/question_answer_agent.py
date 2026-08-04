# component3/question_answer_agent.py
import os, json
from shared.schema import Flag
from component3.retrieval import search_knowledge_base
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient

SYSTEM_PROMPT = """Answer the question using ONLY the retrieved passages below. If they
don't contain the answer, say so plainly instead of guessing.

Return ONLY JSON: {"answer": "...", "answerable": true|false}
"""

class QuestionAnswerAgent:
    def __init__(self):
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )
        self.agent = create_harness_agent(client=client, agent_instructions=SYSTEM_PROMPT, name="QuestionAnswerer")

    async def answer(self, flag: Flag) -> Flag:
        chunks = await search_knowledge_base(flag.resolved_text, top_k=5)
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