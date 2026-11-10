"""Chart follow-ups: multi-metric bar specs and reuse of the stored result."""

import asyncio
from types import SimpleNamespace

import pytest

from data_science import tools

SPEC = {
    "chart_type": "bar",
    "title": "Spend and CPA by channel",
    "data": {
        "categories": ["Display", "Email", "Paid Search"],
        "series": {"Spend": [26185.4, 4144.42, 51923.46], "CPA": [204.57, 3.9, 50.51]},
    },
}


def test_multi_metric_bar_is_valid_and_renders_png():
    assert tools._validate_chart_spec(SPEC)[0]
    png = tools._render_chart_to_bytes(SPEC)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_multi_metric_bar_length_mismatch_rejected():
    bad = {**SPEC, "data": {"categories": ["A", "B"], "series": {"Spend": [1.0]}}}
    assert not tools._validate_chart_spec(bad)[0]


def test_single_series_bar_still_works():
    spec = {"chart_type": "bar", "data": {"categories": ["A", "B"], "values": [1, 2]}}
    assert tools._validate_chart_spec(spec)[0]
    assert tools._render_chart_to_bytes(spec)[:4] == b"\x89PNG"


@pytest.mark.parametrize("text,expected", [
    ("Show me a bar chart of that", True),
    ("Plot this", True),
    ("Visualize the above", True),
    ("Show me a bar chart of spend by channel", False),
    ("Chart that for last year", False),
    ("What was spend last month?", False),
])
def test_chart_followup_detection(text, expected):
    assert tools._is_chart_followup(text) is expected


def test_chart_followup_reuses_stored_result_without_querying(monkeypatch):
    state = {
        "_current_user_request": "Show me a bar chart of that",
        "last_bq_result": "Display 26185.40 ...",
    }
    monkeypatch.setattr(
        "google.adk.tools.agent_tool.AgentTool.run_async",
        lambda *a, **k: pytest.fail("BigQuery agent must not run"),
    )
    out = asyncio.run(tools.call_bigquery_agent("spend and CPA by channel",
                                                 SimpleNamespace(state=state)))
    assert out == "Display 26185.40 ..."
    assert state["last_visualization_result"] == "Display 26185.40 ..."
