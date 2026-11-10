"""Read-only SQL allow-list check, run before any model-written SQL executes.

Fails closed: a query is allowed only if it parses as exactly one
read-only statement and every table it reads is a fully qualified name
on the allow-list.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop,
    exp.Alter, exp.Command, exp.TruncateTable,
)
_BLOCKED_FUNCTIONS = {"EXTERNAL_QUERY", "SESSION_USER", "APPENDS", "CHANGES"}


def _table_name(table: exp.Table) -> str:
    parts = [table.catalog, table.db, table.name]
    if not all(parts):
        return ""
    return ".".join(parts).lower()


def check(sql: str, allowed_tables: set[str]) -> tuple[bool, str]:
    """Return (allowed, reason)."""
    allowed = {t.lower() for t in allowed_tables}
    if not sql or not sql.strip():
        return False, "Empty query."

    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except sqlglot.errors.ParseError as exc:
        return False, f"Query could not be parsed: {str(exc).splitlines()[0]}"

    if len(statements) != 1:
        return False, "Exactly one SQL statement is allowed."
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        return False, "Only read-only SELECT queries are allowed."
    if tree.find(*_WRITE_NODES):
        return False, "Only read-only SELECT queries are allowed."

    for func in tree.find_all(exp.Anonymous, exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).upper()
        if name in _BLOCKED_FUNCTIONS:
            return False, f"Function {name} is not allowed."

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return False, "Table functions are not allowed."
        if not table.db and not table.catalog and table.name.lower() in cte_names:
            continue
        name = _table_name(table)
        if not name:
            return False, (
                f"Table '{table.sql(dialect='bigquery')}' must be fully qualified "
                "as project.dataset.table."
            )
        if name not in allowed:
            return False, f"Table '{name}' is not available for this client."

    return True, ""
