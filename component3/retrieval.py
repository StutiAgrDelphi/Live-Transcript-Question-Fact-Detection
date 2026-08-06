# component3/retrieval.py
"""
Vector search against ai.general_knowledge. This is intentionally separate
from component2/ — the live detection agent has no DB access by design
(see tools/describe_knowledge_base.py); this stage is where DB access
actually lives.

All DB calls go through asyncio.to_thread() — the SQLAlchemy engine here is
synchronous, and calling it directly from an async function blocks the whole
event loop for the duration of the query, which stalls every other flag
that's trying to resolve concurrently. Wrapping it in to_thread() runs the
blocking call on a worker thread instead, so multiple flags can actually
resolve in parallel.
"""
import asyncio
import json
import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, text

from shared.schema import RetrievedChunk

DATABASE_URL = os.environ["DATABASE_URL"]
_engine = create_engine(DATABASE_URL, future=True)

# The similarity floor exists to protect UNSCOPED searches (whole corpus,
# no confirmed document) from returning a confident-looking but wrong match
# from a totally unrelated document. Once a search is SCOPED to one or more
# confirmed documents, that risk is much lower — the only candidates left
# are already known to be about the right subject — so a lower floor there
# trades a little precision for a lot more recall on borderline phrasing.
UNSCOPED_MIN_SIMILARITY = 0.35
SCOPED_MIN_SIMILARITY = 0.15

UNSCOPED_TOP_K = 5
SCOPED_TOP_K = 8  # scoped searches have a much smaller haystack, so pulling more is cheap


async def _embed_query(query: str) -> List[float]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
    )
    resp = await client.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        input=query,
    )
    return resp.data[0].embedding


def _row_to_chunk(content: str, meta_data: dict, similarity: float) -> RetrievedChunk:
    citation = meta_data.get("citation", {})
    return RetrievedChunk(
        content=content,
        document_name=citation.get("document_name") or citation.get("source_file", "Unknown document"),
        document_url=citation.get("sharepoint_document_url") or citation.get("blob_path"),
        page_numbers=citation.get("page_numbers", []),
        similarity=similarity,
    )


def _run_search_query(sql, params):
    """Blocking DB call — always invoke via asyncio.to_thread(), never directly."""
    with _engine.connect() as conn:
        return conn.execute(sql, params).mappings().all()


async def search_knowledge_base(
    query: str,
    top_k: Optional[int] = None,
    document_names: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    query_embedding = await _embed_query(query)

    scoped = bool(document_names)
    effective_top_k = top_k if top_k is not None else (SCOPED_TOP_K if scoped else UNSCOPED_TOP_K)
    min_similarity = SCOPED_MIN_SIMILARITY if scoped else UNSCOPED_MIN_SIMILARITY

    if scoped:
        sql = text("""
            SELECT content, meta_data, 1 - (embedding <=> :qvec) AS similarity
            FROM ai.general_knowledge
            WHERE meta_data->'citation'->>'document_name' = ANY(:doc_names)
            ORDER BY embedding <=> :qvec
            LIMIT :k
        """)
        params = {"qvec": str(query_embedding), "doc_names": document_names, "k": effective_top_k}
    else:
        sql = text("""
            SELECT content, meta_data, 1 - (embedding <=> :qvec) AS similarity
            FROM ai.general_knowledge
            ORDER BY embedding <=> :qvec
            LIMIT :k
        """)
        params = {"qvec": str(query_embedding), "k": effective_top_k}

    rows = await asyncio.to_thread(_run_search_query, sql, params)

    results = [
        _row_to_chunk(
            r["content"],
            r["meta_data"] if isinstance(r["meta_data"], dict) else json.loads(r["meta_data"]),
            r["similarity"],
        )
        for r in rows
    ]
    return [c for c in results if c.similarity >= min_similarity]


def _run_known_documents_query():
    """Blocking DB call — always invoke via asyncio.to_thread(), never directly."""
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
        return conn.execute(sql).mappings().all()


async def list_known_documents() -> List[dict]:
    """Distinct documents in the KB, for the document-lookup resolver.
    Small result set, safe to cache in memory and refresh periodically
    rather than per-request."""
    rows = await asyncio.to_thread(_run_known_documents_query)
    return [dict(r) for r in rows]