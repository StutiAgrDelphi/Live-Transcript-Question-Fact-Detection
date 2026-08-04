"""
Vector search against ai.general_knowledge. This is intentionally separate
from component2/ — the live detection agent has no DB access by design
(see tools/describe_knowledge_base.py); this stage is where DB access
actually lives.
"""
import os
import json
from typing import List
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, text
from agent_framework_openai import OpenAIChatClient  # confirm this client also exposes embeddings;
                                                        # if not, use the openai/azure-openai SDK directly here

DATABASE_URL = os.environ["DATABASE_URL"]
_engine = create_engine(DATABASE_URL, future=True)

_EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

# component3/retrieval.py — replace _embed_query
async def _embed_query(query: str):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
    )

    resp = await client.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        input=query,
    )

    #print("Response:", resp)

    embedding = resp.data[0].embedding

    print("Embedding type:", type(embedding))
    print("Embedding length:", len(embedding))

    return embedding


def _row_to_chunk(content: str, meta_data: dict, similarity: float):
    from shared.schema import RetrievedChunk
    citation = meta_data.get("citation", {})
    return RetrievedChunk(
        content=content,
        document_name=citation.get("document_name") or citation.get("source_file", "Unknown document"),
        document_url=citation.get("sharepoint_document_url") or citation.get("blob_path"),
        page_numbers=citation.get("page_numbers", []),
        similarity=similarity,
    )


async def search_knowledge_base(query: str, top_k: int = 5):
    query_embedding = await _embed_query(query)
    sql = text("""
        SELECT content, meta_data, 1 - (embedding <=> :qvec) AS similarity
        FROM ai.general_knowledge
        ORDER BY embedding <=> :qvec
        LIMIT :k
    """)
    with _engine.connect() as conn:
        rows = conn.execute(sql, {"qvec": str(query_embedding), "k": top_k}).mappings().all()

    results = []
    for r in rows:
        meta = r["meta_data"] if isinstance(r["meta_data"], dict) else json.loads(r["meta_data"])
        results.append(_row_to_chunk(r["content"], meta, r["similarity"]))
    return results


async def list_known_documents() -> List[dict]:
    """Distinct documents in the KB, for the document-lookup resolver.
    Small result set (696 chunks -> far fewer distinct documents), safe to
    cache in memory and refresh periodically rather than per-request."""
    sql = text("""
        SELECT DISTINCT
            meta_data->'citation'->>'document_name' AS document_name,
            COALESCE(
                meta_data->'citation'->>'sharepoint_document_url',
                meta_data->'citation'->>'blob_path'
            ) AS document_url
        FROM ai.general_knowledge
        WHERE meta_data->'citation'->>'document_name' IS NOT NULL
    """)
    with _engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [dict(r) for r in rows]