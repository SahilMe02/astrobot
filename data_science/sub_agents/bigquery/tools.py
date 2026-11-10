"""BigQuery specialist tools: schema loading, NL2SQL prompt, SQL guard.

Tables and business rules come from the active client's data source
(tenancy.data_sources), never from the request text.
"""

from __future__ import annotations

from typing import Any, Optional

import google.auth
from google.adk.tools import BaseTool, ToolContext
from google.cloud import bigquery

from tenancy import data_sources, sql_guard

EXECUTE_SQL_TOOL = "execute_sql"


def get_database_settings() -> dict:
    """Load the schema of the tables the active client may query."""
    source = data_sources.current()
    credentials, _ = google.auth.default()
    client = bigquery.Client(
        project=data_sources._project(),
        credentials=credentials,
    )

    schema_parts: list[str] = []
    for table_ref in source.tables.values():
        table = client.get_table(table_ref)
        columns = "\n".join(
            f"  {f.name} ({f.field_type}, {f.mode})" for f in table.schema
        )
        schema_parts.append(f"-- `{table_ref}`\n{columns}")

    return {
        "client_id": source.client_id,
        "schema": "\n\n".join(schema_parts),
        "tables": sorted(source.tables.values()),
    }


def bigquery_nl2sql(question: str, tool_context: ToolContext) -> dict:
    """Prepare schema-aware instructions for generating BigQuery SQL."""
    source = data_sources.current()
    settings = tool_context.state.get("database_settings", {})
    schema = settings.get("schema", "Schema not loaded.")
    tables = "\n".join(f"- `{t}`" for t in sorted(source.tables.values()))
    rules = "\n".join(f"{i}. {r}" for i, r in enumerate(source.rules, 1))

    return {
        "question": question,
        "schema": schema,
        "instruction": f"""
Generate one read-only BigQuery SELECT query to answer:

{question}

You may ONLY use these tables, always written fully qualified in backticks:
{tables}

Business rules:
{rules}

General rules:
- Do not invent tables or columns; use only the schema below.
- Exactly one SELECT statement (WITH ... SELECT is fine). No DML/DDL.
- Avoid SELECT * unless explicitly required.
- Use SAFE_DIVIDE for ratios.
- Apply date filters when the question names a time period.

Schema:
{schema}
""",
    }


def guard_execute_sql(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict]:
    """before_tool_callback: block any SQL outside the active client's tables.

    Returning a dict skips the real tool call, so a rejected query never
    reaches BigQuery.
    """
    if tool.name != EXECUTE_SQL_TOOL:
        return None

    sql = args.get("query") or args.get("sql") or ""
    try:
        allowed = data_sources.current().allowed_tables
    except data_sources.DataSourceError as exc:
        return {"status": "ERROR", "error_details": str(exc)}

    ok, reason = sql_guard.check(sql, allowed)
    if not ok:
        tool_context.state["last_rejected_sql_reason"] = reason
        return {
            "status": "ERROR",
            "error_details": f"Query blocked by AstroBot data policy: {reason}",
        }
    return None
