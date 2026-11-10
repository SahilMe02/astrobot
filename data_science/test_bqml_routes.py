"""Every deterministic BQML route in call_bqml_agent runs end to end against a
fake BigQuery client, and its SQL targets the active client's model."""

import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest

from data_science import tools
from tenancy import registry
from tenancy.context import client_scope


class FakeJob:
    job_id = "job-1"

    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class FakeClient:
    def __init__(self):
        self.sql = []

    def query(self, sql, **kwargs):
        self.sql.append(sql)
        start = dt.datetime(2026, 11, 8, tzinfo=dt.timezone.utc)
        return FakeJob([
            {
                "forecast_timestamp": start + dt.timedelta(days=i),
                "forecast_value": 100.0 + i,
                "prediction_interval_lower_bound": 90.0 + i,
                "prediction_interval_upper_bound": 110.0 + i,
                "non_seasonal_p": 1, "non_seasonal_d": 1, "non_seasonal_q": 1,
                "AIC": 1.0, "variance": 1.0,
            }
            for i in range(30)
        ])

    def get_model(self, path):
        return SimpleNamespace(_properties={"modelType": "ARIMA_PLUS"}, location="US")


@pytest.fixture
def fake_bq(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    registry.reload()
    client = FakeClient()
    from data_science.sub_agents.bqml import tools as bqml
    monkeypatch.setattr(bqml, "_get_client", lambda: client)
    return client


def _ask(text):
    state = {"_current_user_request": text}
    with client_scope("npi"):
        out = asyncio.run(tools.call_bqml_agent(text, SimpleNamespace(state=state)))
    return out, state


def test_forecast_route_runs_on_client_model(fake_bq):
    out, state = _ask("Forecast conversions for the next 30 days")
    assert "not defined" not in out and "failed" not in out.lower()
    assert "proj-1.astrobot_ml_npi.revenue_forecast" in fake_bq.sql[-1]
    assert state["last_visualization_source"] == "bqml"


def test_evaluation_route_runs_on_client_model(fake_bq):
    out, _ = _ask("Evaluate the forecasting model")
    assert "not defined" not in out
    assert "ML.ARIMA_EVALUATE" in fake_bq.sql[-1]
    assert "astrobot_ml_npi" in fake_bq.sql[-1]


def test_model_status_route_runs(fake_bq):
    out, _ = _ask("What is the status of the forecast model?")
    assert "not defined" not in out


def test_forecast_text_names_the_client_measure(fake_bq):
    out, _ = _ask("Forecast conversions for the next 30 days")
    assert "30-day conversions forecast" in out and "revenue" not in out.lower()
