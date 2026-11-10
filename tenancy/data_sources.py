"""BigQuery sources each client may use, resolved from the registry.

`current()` returns the source for the active client (tenancy.context).
No client -> the public demo source. A client without a valid data block
raises, so data code fails closed instead of falling back to another
client's or the demo data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import registry
from .context import active_client


class DataSourceError(RuntimeError):
    """The active client has no usable data configuration."""


@dataclass(frozen=True)
class Forecast:
    table: str              # fully qualified
    date_expression: str
    value_expression: str
    where: str = ""
    label: str = "value"


@dataclass(frozen=True)
class DataSource:
    client_id: str          # "" for demo
    display_name: str
    tables: dict[str, str]  # short name -> fully qualified
    rules: list[str] = field(default_factory=list)
    bqml_dataset: str = ""  # project.dataset
    forecast: Forecast | None = None
    segmentation: bool = False

    @property
    def allowed_tables(self) -> set[str]:
        return {t.lower() for t in self.tables.values()}


def _project() -> str:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise DataSourceError("GOOGLE_CLOUD_PROJECT is not configured.")
    return project


def _fill(value: str) -> str:
    return value.replace("{project}", _project()) if "{project}" in value else value


def _build(client_id: str, entry: dict) -> DataSource:
    data = entry.get("data") or {}
    tables = {name: _fill(path) for name, path in (data.get("tables") or {}).items()}
    if not tables:
        raise DataSourceError(f"No data sources are configured for '{client_id or 'demo'}'.")

    profiles = registry.raw_config().get("rule_profiles", {})
    rules = list(profiles.get(data.get("rules_profile", ""), [])) + list(data.get("rules", []))

    forecast = None
    if data.get("forecast"):
        f = data["forecast"]
        if f.get("table") not in tables:
            raise DataSourceError(f"Forecast table for '{client_id or 'demo'}' is not in its tables.")
        forecast = Forecast(
            table=tables[f["table"]],
            date_expression=f["date_expression"],
            value_expression=f["value_expression"],
            where=f.get("where", ""),
            label=f.get("label", "value"),
        )

    return DataSource(
        client_id=client_id,
        display_name=entry.get("display_name", client_id),
        tables=tables,
        rules=rules,
        bqml_dataset=_fill(data.get("bqml_dataset", "")),
        forecast=forecast,
        segmentation=bool(data.get("segmentation", False)),
    )


def for_client(client_id: str) -> DataSource:
    client_id = str(client_id or "").strip().lower()
    if not client_id:
        return _build("", registry.raw_config().get("demo") or {})
    entry = registry.clients().get(client_id)
    if entry is None:
        raise DataSourceError(f"Unknown client '{client_id}'.")
    return _build(client_id, entry)


def current() -> DataSource:
    """Data source for the client this code is running for."""
    return for_client(active_client())
