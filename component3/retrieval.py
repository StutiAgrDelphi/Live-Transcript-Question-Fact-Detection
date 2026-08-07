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

ACCESS CONTROL: ai.general_knowledge has Row Level Security enabled, keyed off
two Postgres session settings — app.current_user_id and app.current_user_role.
EVERY function here that touches that table sets both, inside the SAME
transaction as the query (SET LOCAL — cleared automatically when the
transaction ends), because connections are pooled and a setting left on a
connection would otherwise leak onto the next unrelated request that reuses
it. There is no query path in this file that is allowed to skip this.
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


def _set_rls_context(conn, user_id: str, user_role: str):
    """Must be called inside an open transaction (conn.begin()), every time,
    before touching ai.general_knowledge. SET LOCAL so it can never leak onto
    the next pooled request."""
    conn.execute(text("SET LOCAL app.current_user_id = :uid"), {"uid": user_id or ""})
    conn.execute(text("SET LOCAL app.current_user_role = :role"), {"role": user_role or ""})


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


def _run_search_query(sql, params, user_id: str, user_role: str):
    """Blocking DB call — always invoke via asyncio.to_thread(), never directly."""
    with _engine.connect() as conn:
        with conn.begin():  # transaction scope — SET LOCAL is cleared when this ends
            _set_rls_context(conn, user_id, user_role)
            return conn.execute(sql, params).mappings().all()


async def search_knowledge_base(
    user_id: str,
    user_role: str,
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

    # RLS silently strips rows the caller isn't allowed to see — no extra
    # filtering needed here, the policy on ai.general_knowledge does it.
    rows = await asyncio.to_thread(_run_search_query, sql, params, user_id, user_role)

    results = [
        _row_to_chunk(
            r["content"],
            r["meta_data"] if isinstance(r["meta_data"], dict) else json.loads(r["meta_data"]),
            r["similarity"],
        )
        for r in rows
    ]
    return [c for c in results if c.similarity >= min_similarity]


def _run_known_documents_query(user_id: str, user_role: str):
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
        with conn.begin():
            _set_rls_context(conn, user_id, user_role)
            return conn.execute(sql).mappings().all()


async def list_known_documents(user_id: str, user_role: str) -> List[dict]:
    """Distinct documents VISIBLE TO user_id/user_role in the KB (RLS-scoped).
    Callers that need the full, unrestricted index for entity-matching
    purposes should pass the resolver/organizer identity explicitly — this
    function itself never bypasses RLS on its own."""
    rows = await asyncio.to_thread(_run_known_documents_query, user_id, user_role)
    return [dict(r) for r in rows]


def _run_visibility_check(document_name: str, user_id: str, user_role: str) -> bool:
    """Blocking DB call — always invoke via asyncio.to_thread(), never directly.
    Asks: does at least one row for this document exist that user_id/user_role
    can see under RLS? Reuses the exact same policy path as real retrieval,
    so this can never drift out of sync with what search actually returns."""
    sql = text("""
        SELECT EXISTS (
            SELECT 1 FROM ai.general_knowledge
            WHERE meta_data->'citation'->>'document_name' = :doc_name
        ) AS visible
    """)
    with _engine.connect() as conn:
        with conn.begin():
            _set_rls_context(conn, user_id, user_role)
            row = conn.execute(sql, {"doc_name": document_name}).mappings().first()
    return bool(row["visible"]) if row else False


async def is_document_visible(user_id: str, user_role: str, document_name: str) -> bool:
    """RLS-scoped visibility check — used at delivery time to decide whether
    a specific connected viewer is allowed to see a specific document's
    content in a flag that's about to be sent to them."""
    if not document_name:
        return True  # nothing to restrict against
    return await asyncio.to_thread(_run_visibility_check, document_name, user_id, user_role)


def _run_user_role_query(user_id: str) -> str:
    """Blocking DB call — always invoke via asyncio.to_thread(), never directly."""
    sql = text("SELECT role FROM access.users WHERE user_id = :uid")
    with _engine.connect() as conn:
        row = conn.execute(sql, {"uid": user_id}).mappings().first()
    return row["role"] if row else "attendee"


async def get_user_role(user_id: str) -> str:
    """Server-side role lookup — role must NEVER be trusted from the client,
    only user_id is. Unknown users default to the least-privileged role."""
    return await asyncio.to_thread(_run_user_role_query, user_id)