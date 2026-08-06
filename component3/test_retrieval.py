# component3/test_retrieval.py
"""
Run directly: python -m component3.test_retrieval
Isolates the embedding call + DB query from everything else, so any failure
shows its real traceback immediately instead of getting swallowed by the
pipeline's error handling.
"""
import asyncio
from dotenv import load_dotenv
load_dotenv()

from component3.retrieval import search_knowledge_base

async def main():
    results = await search_knowledge_base("Hello Ryan How are you", top_k=3)
    for r in results:
        print(f"{r.similarity:.3f}  {r.document_name}  —  {r.content[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())

    