"""The BigQuery agent must know today's date and report the covered range."""

from datetime import date

from data_science.sub_agents.bigquery.prompts import return_instructions_bigquery
from tenancy import data_sources, registry


def test_instructions_include_today_and_coverage_rule():
    registry.reload()
    text = return_instructions_bigquery(data_sources.for_client("npi"))
    assert f"Today's date is {date.today().isoformat()}." in text
    assert "most recent" in text and "data ends before the requested period" in text
