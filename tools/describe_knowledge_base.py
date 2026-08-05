# tools/describe_knowledge_base.py
"""
Generates knowledge_base_context.md — a description of the database's contents
for the live-meeting detector agent (component2/detector_agent.py) to use as
grounding for its knowledge-base-relevance checks.

Design principles:
- Which documents exist is answered with an exact SQL DISTINCT query — never a
  sample. Sampling is for summarization, not enumeration.
- Where the document identifier lives in a table's schema is discovered per-table
  via one LLM call inspecting real samples — not a hardcoded JSON path, since a
  different table (or a non-document table, e.g. plain tabular data) won't have
  the same shape.
- If no document-identifier field can be confidently found, the script does NOT
  fabricate a "documents present" list — it falls back to a plain schema/content
  description instead. Accuracy over completeness.
"""
import os
import sys
import json
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.engine import Engine
from openai import AsyncAzureOpenAI

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SAMPLE_ROWS_PER_TABLE = 12          # for the prose "what's in this table" summary
DISTINCT_VALUES_LIMIT = 500         # cap for exhaustive document-name enumeration
CATEGORY_COLUMN_HINTS = {           # flat-column hints (cheap, no LLM call needed)
    "category", "type", "tag", "topic", "label", "kind", "status",
    "source", "document", "doc_name", "doc_id", "title", "citation",
    "file", "filename",
}
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base_context.md")
SKIP_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast"}


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def _build_engine() -> Engine:
    db_url = os.environ["DATABASE_URL"]
    return create_engine(db_url)


def _build_llm_client() -> Tuple[AsyncAzureOpenAI, str]:
    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    return client, deployment


async def _llm(client: AsyncAzureOpenAI, deployment: str, prompt: str, temperature: float = 0.0) -> str:
    resp = await client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

def _discover_tables(engine: Engine, only: Optional[List[str]]) -> List[Tuple[Optional[str], str]]:
    if only:
        tables = []
        for spec in only:
            if "." in spec:
                schema, table_name = spec.split(".", 1)
            else:
                schema, table_name = None, spec
            tables.append((schema, table_name))
        return tables

    insp = sa_inspect(engine)
    try:
        schemas = insp.get_schema_names()
    except Exception:
        schemas = [None]

    tables: List[Tuple[Optional[str], str]] = []
    for schema in schemas:
        if schema in SKIP_SCHEMAS:
            continue
        try:
            for table_name in insp.get_table_names(schema=schema):
                tables.append((schema, table_name))
            for view_name in insp.get_view_names(schema=schema):
                tables.append((schema, view_name))
        except Exception as e:
            log.warning(f"Could not list tables in schema '{schema}': {e}")
    return tables


def _qualified(schema: Optional[str], table_name: str) -> str:
    return f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'


def _get_columns(engine: Engine, schema: Optional[str], table_name: str) -> List[Dict[str, Any]]:
    insp = sa_inspect(engine)
    try:
        return insp.get_columns(table_name, schema=schema)
    except Exception as e:
        log.warning(f"Could not read columns for {_qualified(schema, table_name)}: {e}")
        return []


def _is_json_type(col: Dict[str, Any]) -> bool:
    type_name = str(col.get("type", "")).upper()
    return "JSON" in type_name


def _row_count(engine: Engine, schema: Optional[str], table_name: str) -> Optional[int]:
    qualified = _qualified(schema, table_name)
    try:
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT COUNT(*) FROM {qualified}")).scalar()
    except Exception as e:
        log.warning(f"Could not count rows for {qualified}: {e}")
        return None


def _sample_rows(engine: Engine, schema: Optional[str], table_name: str, limit: int) -> List[Dict[str, Any]]:
    qualified = _qualified(schema, table_name)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {qualified} ORDER BY RANDOM() LIMIT :lim"), {"lim": limit}
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"Could not sample rows for {qualified}: {e}")
        return []


def _fetch_distinct_flat_column(engine: Engine, schema: Optional[str], table_name: str, col_name: str) -> List[Any]:
    qualified = _qualified(schema, table_name)
    query = text(f'SELECT DISTINCT "{col_name}" FROM {qualified} WHERE "{col_name}" IS NOT NULL LIMIT :lim')
    with engine.connect() as conn:
        rows = conn.execute(query, {"lim": DISTINCT_VALUES_LIMIT}).fetchall()
    return [r[0] for r in rows]


def _json_path_to_sql_expr(json_col: str, dot_path: str) -> str:
    parts = dot_path.split(".")
    expr = f'"{json_col}"'
    for p in parts[:-1]:
        expr += f"->'{p}'"
    expr += f"->>'{parts[-1]}'"
    return expr


def _fetch_distinct_json_path(engine: Engine, schema: Optional[str], table_name: str, json_col: str, dot_path: str) -> List[str]:
    qualified = _qualified(schema, table_name)
    expr = _json_path_to_sql_expr(json_col, dot_path)
    query = text(f"SELECT DISTINCT {expr} AS v FROM {qualified} WHERE {expr} IS NOT NULL LIMIT :lim")
    with engine.connect() as conn:
        rows = conn.execute(query, {"lim": DISTINCT_VALUES_LIMIT}).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# LLM-assisted discovery of a document-identifier field (no hardcoded path)
# ---------------------------------------------------------------------------

async def _discover_json_document_id_path(
    client: AsyncAzureOpenAI, deployment: str, json_col: str, samples: List[Any]
) -> Optional[str]:
    """
    Inspects a handful of raw JSON values from one JSON/JSONB column and asks the
    LLM whether there's a nested field that identifies which SOURCE DOCUMENT a row
    belongs to (a value that would repeat across many rows/chunks of the same
    document — e.g. a filename, title, or citation field). Returns a dot-path
    ("citation.document_name") if confident, or None if no such field is evident.
    This is deliberately per-table and per-run — it never assumes the answer from
    a previous table or a previous version of this script.
    """
    cleaned = [s for s in samples if s]
    if not cleaned:
        return None
    prompt = f"""You are inspecting sample JSON values from a database column called
"{json_col}" to find a nested field that identifies which SOURCE DOCUMENT or FILE a
row came from — a value that would repeat across many rows belonging to the same
document (e.g. a filename, document title, or citation field). This is NOT looking
for a unique per-row ID; it's looking for a field with LOW cardinality relative to
the table, shared across chunks of the same source.

Sample values (each is one row's JSON, may be truncated):
{json.dumps(cleaned[:8], indent=2, default=str)[:4000]}

If you can identify such a field with confidence, respond with ONLY its dot-path
from the top of the JSON (example format: citation.document_name), nothing else —
no explanation, no punctuation around it.
If no such field is evident from these samples, respond with exactly: NONE
Do not guess a path you are not confident about — respond NONE instead."""
    try:
        answer = await _llm(client, deployment, prompt)
    except Exception as e:
        log.warning(f"LLM document-id-path discovery failed for column '{json_col}': {e}")
        return None
    answer = answer.strip().strip('"').strip("'")
    if not answer or answer.upper() == "NONE":
        return None
    return answer


async def _find_document_identifiers(
    engine: Engine, client: AsyncAzureOpenAI, deployment: str,
    schema: Optional[str], table_name: str, columns: List[Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Returns {label: [complete list of distinct document identifiers]} for every
    identifier field found in this table — flat columns via cheap hint-matching,
    JSON-nested fields via one LLM call per JSON column. Returns {} if nothing
    confident is found, meaning: don't claim this table is a document corpus.
    """
    found: Dict[str, List[str]] = {}

    # Cheap path: flat columns matching known hint keywords.
    for col in columns:
        lower = col["name"].lower()
        if any(hint in lower for hint in CATEGORY_COLUMN_HINTS) and not _is_json_type(col):
            try:
                vals = _fetch_distinct_flat_column(engine, schema, table_name, col["name"])
                if vals and 1 < len(vals) <= DISTINCT_VALUES_LIMIT:
                    found[col["name"]] = [str(v) for v in vals]
            except Exception as e:
                log.warning(f"Distinct fetch failed for column '{col['name']}': {e}")

    # LLM-assisted path: JSON/JSONB columns, discovered fresh each run.
    for col in columns:
        if not _is_json_type(col):
            continue
        json_samples = [row.get(col["name"]) for row in sample_rows if row.get(col["name"])]
        path = await _discover_json_document_id_path(client, deployment, col["name"], json_samples)
        if not path:
            continue
        try:
            vals = _fetch_distinct_json_path(engine, schema, table_name, col["name"], path)
            if vals:
                found[f"{col['name']}.{path}"] = [str(v) for v in vals]
                log.info(f"  Found document identifier: {col['name']} -> {path} ({len(vals)} distinct values)")
        except Exception as e:
            log.warning(f"JSON-path distinct fetch failed for {col['name']}.{path}: {e}")

    return found


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(
    schema: Optional[str], table_name: str, columns: List[Dict[str, Any]],
    row_count: Optional[int], sample_rows: List[Dict[str, Any]],
    document_identifiers: Dict[str, List[str]],
) -> str:
    col_summary = "\n".join(f"- {c['name']} ({c['type']})" for c in columns)
    sample_json = json.dumps(sample_rows[:8], indent=2, default=str)[:6000]

    if document_identifiers:
        # Flatten to one exhaustive, deduplicated list — this is the ground truth,
        # not a sample. Every one of these MUST appear in the output.
        all_names: List[str] = []
        for names in document_identifiers.values():
            all_names.extend(names)
        all_names = sorted(set(all_names))
        names_block = "\n".join(f"- {n}" for n in all_names)

        return f"""You are writing a short reference section of knowledge_base_context.md,
describing table "{table_name}" (schema: {schema or 'default'}) for a live-meeting
detection agent. This table holds chunks belonging to distinct source documents.

Columns:
{col_summary}

Approximate row count: {row_count if row_count is not None else 'unknown'}

Sample row content (for flavor/topics only — NOT the full document list):
{sample_json}

COMPLETE, EXACT list of every distinct document identifier actually present in this
table (from a direct database query, not a sample — this is ground truth):
{names_block}

Write two things:

1. "### What this table contains" — 2-4 sentences on what kind of documents these
   are overall and what topics/domains they span. Keep it general, not exhaustive.

2. "### Documents and sources present" — EVERY single name in the exact list above
   MUST appear somewhere in this section, verbatim (do not paraphrase, shorten, or
   drop any of them — a meeting agent needs the real names to match speech against).
   Group them sensibly (e.g. by naming pattern, topic, or document family) with a
   short one-line description per GROUP rather than per document, so the list stays
   readable even with many documents. If two names look like versions of the same
   underlying document (e.g. "X" and "X_v2"), you may note that in one line rather
   than describing both separately — but both names must still literally appear in
   the text so they can be matched against spoken references to either.

Do not invent documents, topics, or names beyond what's given above. Do not add
a "Trigger phrases" or "document_lookup examples" section — that's handled
separately. Output only the two sections above, in Markdown."""

    else:
        return f"""You are writing a short reference section of knowledge_base_context.md,
describing table "{table_name}" (schema: {schema or 'default'}) for a live-meeting
detection agent. No confident per-document identifier field was found in this
table's structure — treat it as general tabular/reference data, NOT a document
corpus. Do not invent a document list.

Columns:
{col_summary}

Approximate row count: {row_count if row_count is not None else 'unknown'}

Sample row content:
{sample_json}

Write one section, "### What this table contains" — 3-5 sentences describing, in
plain terms, what kind of data this table actually holds (based only on the columns
and samples above), and what a meeting participant might plausibly be discussing if
they reference this data. Stay grounded in what's actually shown — do not guess at
structure or content beyond the samples given."""


async def _build_vocab_and_examples(
    client: AsyncAzureOpenAI, deployment: str, table_name: str,
    description_section: str, has_documents: bool,
) -> str:
    kind = "specific documents" if has_documents else "topics/records"
    prompt = f"""Based on this description of a knowledge base table:

{description_section}

Write two short Markdown sections:

1. "### Trigger phrases and vocabulary" — a single comma-separated line of terms,
   names, and phrases a meeting participant would plausibly say out loud that
   relate to this data. Use real names/terms from the description above only.

2. "### What to flag as document_lookup" — 4-6 example spoken sentences someone
   might say to reference {kind} in this table (natural speech, not commands).
   {"Reference real document names from the description above." if has_documents else "Only include this section if it genuinely makes sense to 'look up' something specific here — for plain data tables, it's fine to write one line noting that document_lookup likely does not apply to this table."}

Output only these two sections."""
    return await _llm(client, deployment, prompt)


# ---------------------------------------------------------------------------
# Per-table pipeline
# ---------------------------------------------------------------------------

async def _describe_table(
    engine: Engine, client: AsyncAzureOpenAI, deployment: str,
    schema: Optional[str], table_name: str,
) -> Optional[str]:
    qualified = _qualified(schema, table_name)
    log.info(f"Describing {qualified}...")

    columns = _get_columns(engine, schema, table_name)
    if not columns:
        return None

    row_count = _row_count(engine, schema, table_name)
    sample_rows = _sample_rows(engine, schema, table_name, SAMPLE_ROWS_PER_TABLE)
    if not sample_rows:
        log.warning(f"  No rows found in {qualified}, skipping.")
        return None

    document_identifiers = await _find_document_identifiers(
        engine, client, deployment, schema, table_name, columns, sample_rows
    )

    description_prompt = _build_prompt(schema, table_name, columns, row_count, sample_rows, document_identifiers)
    description_section = await _llm(client, deployment, description_prompt)

    vocab_section = await _build_vocab_and_examples(
        client, deployment, table_name, description_section, bool(document_identifiers)
    )

    heading = f"## {schema + '.' if schema else ''}{table_name}"
    return f"{heading}\n\n{description_section}\n\n{vocab_section}\n"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Generate knowledge_base_context.md")
    parser.add_argument(
        "--table", action="append", default=None, metavar="schema.table",
        help="Restrict to specific table(s), repeatable. Omit to auto-discover the whole DB.",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()
    print("=" * 60)
    print("Knowledge Base Description Generator")
    print("=" * 60)

    engine = _build_engine()
    client, deployment = _build_llm_client()

    tables = _discover_tables(engine, args.table)
    if not tables:
        print("No tables found. Check DATABASE_URL and connection permissions.")
        return

    print(f"\nFound {len(tables)} table(s)/view(s). Generating descriptions...\n")

    sections: List[str] = []
    for schema, table_name in tables:
        try:
            section = await _describe_table(engine, client, deployment, schema, table_name)
            if section:
                sections.append(section)
        except Exception as e:
            log.error(f"Failed to describe {_qualified(schema, table_name)}: {e}")

    if not sections:
        print("No table descriptions were generated. Nothing written.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    output = f"# Knowledge Base Context\n\nGenerated: {timestamp}\n\n" + "\n".join(sections)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nWrote {OUTPUT_PATH} ({len(sections)} table section(s)).")


if __name__ == "__main__":
    asyncio.run(main())