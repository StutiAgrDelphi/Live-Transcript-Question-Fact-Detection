# tools/describe_knowledge_base.py
"""
Standalone offline tool — run manually to (re)generate knowledge_base_context.md.

Usage:
    python -m tools.describe_knowledge_base

What it does:
    1. Connects to the database via DATABASE_URL from .env
    2. Introspects every table it finds (any schema, any table name)
    3. Pulls a modest random sample of rows + distinct values for category-like columns
    4. Calls the LLM once per table to write a semantic description
    5. Writes knowledge_base_context.md at the project root

Output file (knowledge_base_context.md) is the ONLY thing detector_agent.py ever sees
from this database — the live detection agent has no direct DB access.

This script does not import from component1/, component2/, or api/.
Nothing in those packages should import from this script either.
"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tunable constants — adjust here, not scattered through the code
# ---------------------------------------------------------------------------

# Number of random rows sampled per table to ground the LLM description.
# Keep this SMALL. This script is for understanding content, not exporting data.
# PRIVACY NOTE: If this script is ever run on a schedule, or if any table contains
# genuinely sensitive personal data (PII, medical, financial), review whether
# sampling real rows is appropriate before automating. For a manually-run, one-time
# description pass it is acceptable — flag and review if that assumption changes.
SAMPLE_ROWS_PER_TABLE = 10

# Max distinct values fetched for category/type/tag-style columns.
DISTINCT_VALUES_LIMIT = 50

# Column name substrings that trigger a DISTINCT value fetch (case-insensitive).
CATEGORY_COLUMN_HINTS = {"category", "type", "tag", "topic", "label", "kind", "status"}

# Where to write the output.
OUTPUT_FILE = Path(__file__).parent.parent / "knowledge_base_context.md"

# ---------------------------------------------------------------------------
# DB and LLM setup
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# DATABASE_URL is expected to point to a READ-ONLY database role where possible.
# Using a read-only role prevents this script from accidentally modifying data.
# Document the expectation here; we do not enforce it in code.
DATABASE_URL: str = os.environ["DATABASE_URL"]


def _build_engine():
    from sqlalchemy import create_engine
    return create_engine(DATABASE_URL, future=True)


def _build_llm_client():
    """Direct OpenAIChatClient — no harness, just one prompt-in / text-out per table."""
    from agent_framework_openai import OpenAIChatClient
    return OpenAIChatClient(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="preview",
    )


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

def _is_category_column(col_name: str) -> bool:
    lower = col_name.lower()
    return any(hint in lower for hint in CATEGORY_COLUMN_HINTS)


def _introspect_table(engine, schema: Optional[str], table_name: str) -> Dict[str, Any]:
    """
    Returns a dict with:
      columns       — list of column info dicts
      pk_columns    — list of primary key column names
      fk_info       — list of FK descriptions
      table_comment — string or None
    """
    from sqlalchemy import inspect as sa_inspect, text

    insp = sa_inspect(engine)
    qualified = f"{schema}.{table_name}" if schema else table_name

    columns = []
    for col in insp.get_columns(table_name, schema=schema):
        columns.append(
            {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "default": str(col.get("default", "")),
                "comment": col.get("comment"),
            }
        )

    pk = insp.get_pk_constraint(table_name, schema=schema)
    pk_cols = pk.get("constrained_columns", []) if pk else []

    fks = []
    for fk in insp.get_foreign_keys(table_name, schema=schema):
        fks.append(
            f"{fk['constrained_columns']} → "
            f"{fk.get('referred_schema','')}.{fk['referred_table']}"
            f"({fk['referred_columns']})"
        )

    # Table-level comment — not all backends support this; ignore errors silently
    table_comment = None
    try:
        table_comment = insp.get_table_comment(table_name, schema=schema).get("text")
    except Exception:
        pass

    return {
        "qualified_name": qualified,
        "columns": columns,
        "pk_columns": pk_cols,
        "fk_info": fks,
        "table_comment": table_comment,
    }


def _fetch_distinct_category_values(
    engine, schema: Optional[str], table_name: str, col_name: str
) -> List[Any]:
    from sqlalchemy import text

    qualified = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    query = text(
        f'SELECT DISTINCT "{col_name}" FROM {qualified} '
        f"WHERE \"{col_name}\" IS NOT NULL LIMIT :lim"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"lim": DISTINCT_VALUES_LIMIT}).fetchall()
    return [r[0] for r in rows]


def _fetch_sample_rows(
    engine, schema: Optional[str], table_name: str
) -> Tuple[List[str], List[List[Any]]]:
    """
    Returns (column_names, rows).

    NOTE: ORDER BY RANDOM() can be slow on very large tables — acceptable tradeoff
    for a manually-run, one-time script. If tables ever grow to millions of rows,
    replace with tablesample or an indexed random approach.
    """
    from sqlalchemy import text

    qualified = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    query = text(
        f"SELECT * FROM {qualified} ORDER BY RANDOM() LIMIT :lim"
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"lim": SAMPLE_ROWS_PER_TABLE})
        col_names = list(result.keys())
        rows = [list(r) for r in result.fetchall()]
    return col_names, rows


# ---------------------------------------------------------------------------
# LLM description generation
# ---------------------------------------------------------------------------

def _build_prompt(
    schema_info: Dict[str, Any],
    distinct_by_column: Dict[str, List[Any]],
    col_names: List[str],
    sample_rows: List[List[Any]],
) -> str:
    table_name = schema_info["qualified_name"]

    col_lines = []
    for col in schema_info["columns"]:
        pk_marker = " [PK]" if col["name"] in schema_info["pk_columns"] else ""
        nullable = "" if col["nullable"] else " NOT NULL"
        comment = f"  -- {col['comment']}" if col.get("comment") else ""
        col_lines.append(
            f"  {col['name']}: {col['type']}{nullable}{pk_marker}{comment}"
        )

    fk_section = ""
    if schema_info["fk_info"]:
        fk_section = "\nForeign keys:\n" + "\n".join(
            f"  {fk}" for fk in schema_info["fk_info"]
        )

    table_comment_section = ""
    if schema_info["table_comment"]:
        table_comment_section = f"\nTable comment: {schema_info['table_comment']}"

    distinct_section = ""
    if distinct_by_column:
        parts = []
        for col, vals in distinct_by_column.items():
            vals_str = ", ".join(repr(v) for v in vals[:30])
            if len(vals) > 30:
                vals_str += f" ... ({len(vals)} total shown)"
            parts.append(f"  {col}: [{vals_str}]")
        distinct_section = "\nDistinct values in category-like columns:\n" + "\n".join(parts)

    sample_section = ""
    if sample_rows:
        header = " | ".join(col_names)
        divider = "-" * min(len(header), 120)
        row_lines = []
        for row in sample_rows:
            # Truncate individual cell values to keep the prompt sane
            cells = [str(v)[:200] if v is not None else "NULL" for v in row]
            row_lines.append(" | ".join(cells))
        sample_section = (
            f"\nRandom sample ({len(sample_rows)} rows):\n"
            + divider + "\n"
            + header + "\n"
            + divider + "\n"
            + "\n".join(row_lines)
        )
    else:
        sample_section = "\n(Table appears to be empty — no rows sampled.)"

    return f"""You are writing a semantic description of a database table to help a meeting-detection AI understand what kind of information this database contains.

Below is everything we know about the table `{table_name}`:

=== SCHEMA ===
Columns:
{chr(10).join(col_lines)}{fk_section}{table_comment_section}
{distinct_section}
{sample_section}

=== TASK ===
You are writing a reference document for a real-time meeting detection AI agent.
This agent monitors live meeting transcripts and must instantly decide whether
something spoken relates to content in this database. Your output will be injected
directly into that agent's system prompt.

If the sampled rows contain chunk names or content IDs (e.g. 'Nvidia_2025_Annual_Report_chunk_15'), 
extract the base document names from those identifiers and list them explicitly in 'Documents and 
sources present' even if the chunk content itself does not clearly reveal the topic

Write output in EXACTLY this structure (use these exact headings):

### What this knowledge base contains
One paragraph: what kind of data this table holds, what domain it serves, what
the documents/records are actually about. Be specific — name the actual documents,
companies, and policies you see in the sample. Do NOT use vague phrases like
"a variety of topics."

### Documents and sources present
A bullet list of every distinct document, company, policy, or named source
identifiable from the sampled content. Format each as:
- **[Document/Source Name]** — [1-sentence description of what it covers]

Real meeting speech paraphrases documents loosely — people will not quote column
names or exact clause titles out loud. For "Trigger phrases and vocabulary",
include not just exact terms from the data, but also natural spoken paraphrases
someone would plausibly use to refer to the same concept in conversation.

### Trigger phrases and vocabulary
A flat list of specific terms, names, numbers, and phrases that would appear
in a meeting conversation when someone is discussing content from this database.
These are the signals the agent listens for. Include: company names, document
names, policy names, specific financial figures if present, specific clause names,
product names, regulation names. Format as a comma-separated list.

### What to flag as document_lookup
Specific examples of phrases a meeting participant might say that should be
classified as a document_lookup pointing to content in this database. Give
5-8 concrete examples drawn from the actual content in the sample.
Format as a bullet list of quoted phrases.

Do NOT write prose paragraphs beyond the "What this knowledge base contains"
section. The rest must be structured lists — the agent needs to scan this quickly
mid-classification.
"""


async def _call_llm(client, prompt: str) -> str:
    """Single prompt-in / text-out call — no harness, no multi-turn loop."""
    from agent_framework import Message, ChatOptions

    response = await client.get_response(
        messages=[Message(role="user", contents=prompt)],
        options=ChatOptions(),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Per-table orchestration
# ---------------------------------------------------------------------------

async def _describe_table(
    engine, client, schema: Optional[str], table_name: str
) -> Optional[str]:
    """Returns the LLM-written description string, or None on failure."""
    qualified = f"{schema}.{table_name}" if schema else table_name
    print(f"  Processing table: {qualified} ...", end=" ", flush=True)

    try:
        schema_info = _introspect_table(engine, schema, table_name)
    except Exception as e:
        log.warning(f"Schema introspection failed for {qualified}: {e}")
        return None

    # Distinct values for category-like columns
    distinct_by_column: Dict[str, List[Any]] = {}
    for col in schema_info["columns"]:
        if _is_category_column(col["name"]):
            try:
                vals = _fetch_distinct_category_values(engine, schema, table_name, col["name"])
                if vals:
                    distinct_by_column[col["name"]] = vals
            except Exception as e:
                log.warning(f"Distinct fetch failed for {qualified}.{col['name']}: {e}")

    # Random row sample
    try:
        col_names, sample_rows = _fetch_sample_rows(engine, schema, table_name)
        print(f"sampled {len(sample_rows)} rows.", flush=True)
    except Exception as e:
        log.warning(f"Row sampling failed for {qualified}: {e}")
        col_names, sample_rows = [], []
        print("sampling failed — continuing with schema only.", flush=True)

    prompt = _build_prompt(schema_info, distinct_by_column, col_names, sample_rows)

    try:
        description = await _call_llm(client, prompt)
    except Exception as e:
        log.warning(f"LLM call failed for {qualified}: {e}")
        return None

    return description


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Generate knowledge_base_context.md")
    parser.add_argument(
        "--table",
        action="append",
        default=None,
        metavar="schema.table",
        help="Restrict to specific table(s), repeatable. Omit to auto-discover the whole DB.",
    )
    return parser.parse_args()


async def main():
    from sqlalchemy import inspect as sa_inspect

    args = _parse_args()
    print("=" * 60)
    print("Knowledge Base Description Generator")
    print("=" * 60)

    engine = _build_engine()
    client = _build_llm_client()
    insp = sa_inspect(engine)

    if args.table:
        tables_found = []
        for spec in args.table:
            if "." in spec:
                schema, table_name = spec.split(".", 1)
            else:
                schema, table_name = None, spec
            tables_found.append((schema, table_name))
    else:
        SKIP_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast"}
        tables_found = []
        try:
            schemas = insp.get_schema_names()
        except Exception:
            schemas = [None]

        for schema in schemas:
            if schema in SKIP_SCHEMAS:
                continue
            try:
                for table_name in insp.get_table_names(schema=schema):
                    tables_found.append((schema, table_name))
                for view_name in insp.get_view_names(schema=schema):
                    tables_found.append((schema, view_name))
            except Exception as e:
                log.warning(f"Could not list tables in schema '{schema}': {e}")

    if not tables_found:
        print("No tables found. Check DATABASE_URL and connection permissions.")
        return

    print(f"\nFound {len(tables_found)} table(s)/view(s). Generating descriptions...\n")

    sections: List[str] = []
    successes = 0
    failures = 0

    for schema, table_name in tables_found:
        description = await _describe_table(engine, client, schema, table_name)
        qualified = f"{schema}.{table_name}" if schema else table_name
        if description:
            sections.append(f"## {qualified}\n\n{description}")
            successes += 1
        else:
            sections.append(
                f"## {qualified}\n\n"
                f"_Description could not be generated (see warnings above)._"
            )
            failures += 1

    # Write output
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    output = (
        f"# Knowledge Base Context\n\n"
        f"Generated: {timestamp}\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n"
    )

    OUTPUT_FILE.write_text(output, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Done. {successes} table(s) described, {failures} failed.")
    print(f"Output written to: {OUTPUT_FILE.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())