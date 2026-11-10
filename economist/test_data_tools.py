"""Economist data tools: real API response shapes are parsed correctly,
failures and missing keys return errors (never numbers), no live HTTP."""

from types import SimpleNamespace

import pytest

from economist import data_tools as dt


class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise dt.requests.HTTPError(f"{self.status_code}")


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    dt._secret.cache_clear()
    monkeypatch.setenv("ASTROBOT_FRED_API_KEY", "k")
    monkeypatch.setenv("ASTROBOT_NOAA_API_TOKEN", "t")
    monkeypatch.delenv("ASTROBOT_BLS_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    yield
    dt._secret.cache_clear()


def test_fred_parses_and_summarises(monkeypatch):
    body = {"observations": [
        {"date": "2026-07-01", "value": "60.0"},
        {"date": "2026-08-01", "value": "."},      # FRED's missing-value marker
        {"date": "2026-09-01", "value": "54.0"},
    ]}
    monkeypatch.setattr(dt.requests, "get", lambda *a, **k: FakeResponse(body))
    out = dt.get_fred_series("UMCSENT", "2026-07-01", "2026-09-30")
    assert out["series_id"] == "UMCSENT" and "sentiment" in out["description"]
    assert out["summary"] == {"first": 60.0, "last": 54.0, "min": 54.0, "max": 60.0,
                              "change_pct": -10.0, "points": 2}


def test_fred_missing_key_is_an_error_not_data(monkeypatch):
    monkeypatch.delenv("ASTROBOT_FRED_API_KEY")
    monkeypatch.setattr(dt.requests, "get", lambda *a, **k: pytest.fail("no HTTP without key"))
    out = dt.get_fred_series("UMCSENT", "2026-01-01", "2026-02-01")
    assert "not configured" in out["error"] and "summary" not in out


def test_fred_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(dt.requests, "get", lambda *a, **k: pytest.fail("no HTTP"))
    assert "error" in dt.get_fred_series("UMCSENT&api_key=x", "2026-01-01", "2026-02-01")
    assert "error" in dt.get_fred_series("UMCSENT", "last month", "2026-02-01")


def test_http_failure_is_reported(monkeypatch):
    monkeypatch.setattr(dt.requests, "get", lambda *a, **k: FakeResponse({}, 500))
    out = dt.get_fred_series("FEDFUNDS", "2026-01-01", "2026-02-01")
    assert "request failed" in out["error"]


def test_bls_parses_in_date_order(monkeypatch):
    body = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [{"data": [
        {"year": "2026", "periodName": "September", "value": "4.3"},
        {"year": "2026", "periodName": "August", "value": "4.1"},
    ]}]}}
    sent = {}
    monkeypatch.setattr(dt.requests, "post",
                        lambda url, json, timeout: sent.update(json) or FakeResponse(body))
    out = dt.get_bls_series("LNS14000000", "2026", "2026")
    assert [p["value"] for p in out["observations"]] == [4.1, 4.3]
    assert "registrationkey" not in sent  # optional key absent


def test_bls_failure_status(monkeypatch):
    monkeypatch.setattr(dt.requests, "post", lambda *a, **k: FakeResponse(
        {"status": "REQUEST_NOT_PROCESSED", "message": ["Series does not exist"]}))
    assert "does not exist" in dt.get_bls_series("BAD1", "2026", "2026")["error"]


def test_noaa_groups_by_day(monkeypatch):
    body = {"results": [
        {"date": "2026-11-01T00:00:00", "datatype": "TMAX", "value": 80},
        {"date": "2026-11-01T00:00:00", "datatype": "TMIN", "value": 60},
        {"date": "2026-11-02T00:00:00", "datatype": "PRCP", "value": 0.1},
    ]}
    monkeypatch.setattr(dt.requests, "get", lambda *a, **k: FakeResponse(body))
    out = dt.get_noaa_daily_weather("GHCND:USW00023169", "2026-11-01", "2026-11-02")
    assert out["days"] == [{"date": "2026-11-01", "tmax": 80, "tmin": 60},
                           {"date": "2026-11-02", "prcp": 0.1}]


def test_holidays_national_plus_state(monkeypatch):
    body = [
        {"date": "2026-11-26", "name": "Thanksgiving Day", "global": True, "counties": None},
        {"date": "2026-10-30", "name": "Nevada Day", "global": False, "counties": ["US-NV"]},
        {"date": "2026-03-02", "name": "Texas Independence Day", "global": False, "counties": ["US-TX"]},
    ]
    monkeypatch.setattr(dt.requests, "get", lambda *a, **k: FakeResponse(body))
    out = dt.get_us_holidays("2026", "nv")
    assert [h["name"] for h in out["federal"]] == ["Thanksgiving Day"]
    assert [h["name"] for h in out["state_specific"]] == ["Nevada Day"]
    assert "incomplete" in out["note"] or "may be missing" in out["note"]


def test_economist_wiring():
    from google.adk.tools.google_search_tool import GoogleSearchTool

    from economist.agent import root_agent

    search = [t for t in root_agent.tools if isinstance(t, GoogleSearchTool)]
    assert search and search[0].bypass_multi_tools_limit
    names = {getattr(t, "__name__", "") for t in root_agent.tools}
    assert {"get_fred_series", "get_bls_series", "get_noaa_daily_weather",
            "get_us_holidays"} <= names
    text = root_agent.instruction(SimpleNamespace())
    assert text.startswith("Today's date is ") and "get_fred_series" in text
