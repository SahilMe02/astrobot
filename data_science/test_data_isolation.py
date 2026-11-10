"""Data Science isolation: per-client sources, SQL guard wiring, BQML targets,
and the training approval gate. No BigQuery calls are made."""

import json
from types import SimpleNamespace

import pytest

from tenancy import data_sources, registry
from tenancy.context import client_scope

PROJECT = "proj-1"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", PROJECT)
    monkeypatch.delenv("ASTROBOT_CLIENTS_FILE", raising=False)
    registry.reload()
    yield
    registry.reload()


# ------------------------------------------------------------
# Data sources
# ------------------------------------------------------------

def test_each_client_gets_only_its_own_tables():
    for client in ("npi", "venetian", "winndixie"):
        src = data_sources.for_client(client)
        assert src.allowed_tables == {f"{PROJECT}.astrobot_{client}.performance"}
        assert src.bqml_dataset == f"{PROJECT}.astrobot_ml_{client}"


def test_no_client_uses_demo_only():
    src = data_sources.for_client("")
    assert all(t.startswith("bigquery-public-data.thelook_ecommerce.") for t in src.allowed_tables)


def test_unknown_client_fails_closed():
    with pytest.raises(data_sources.DataSourceError):
        data_sources.for_client("acme")


def test_current_follows_client_scope():
    with client_scope("venetian"):
        assert data_sources.current().client_id == "venetian"
    assert data_sources.current().client_id == ""


# ------------------------------------------------------------
# SQL guard wiring on the BigQuery agent
# ------------------------------------------------------------

def _guard():
    from data_science.sub_agents.bigquery import agent, tools
    assert agent.bigquery_agent.before_tool_callback is tools.guard_execute_sql
    return tools.guard_execute_sql


def _call(guard, sql):
    tool = SimpleNamespace(name="execute_sql")
    return guard(tool, {"query": sql}, SimpleNamespace(state={}))


def test_guard_blocks_other_client_table():
    guard = _guard()
    with client_scope("npi"):
        result = _call(guard, f"SELECT * FROM `{PROJECT}.astrobot_venetian.performance`")
    assert result["status"] == "ERROR" and "not available" in result["error_details"]


def test_guard_allows_own_table():
    guard = _guard()
    with client_scope("npi"):
        assert _call(guard, f"SELECT SUM(Cost) FROM `{PROJECT}.astrobot_npi.performance`") is None


def test_guard_blocks_demo_tables_inside_client_session():
    guard = _guard()
    with client_scope("npi"):
        result = _call(guard, "SELECT 1 FROM `bigquery-public-data.thelook_ecommerce.orders`")
    assert result["status"] == "ERROR"


def test_guard_blocks_client_tables_in_demo_session():
    guard = _guard()
    result = _call(guard, f"SELECT 1 FROM `{PROJECT}.astrobot_npi.performance`")
    assert result["status"] == "ERROR"


def test_guard_ignores_other_tools():
    guard = _guard()
    assert guard(SimpleNamespace(name="bigquery_nl2sql"), {}, SimpleNamespace(state={})) is None


# ------------------------------------------------------------
# BQML targets follow the client
# ------------------------------------------------------------

def test_bqml_sql_targets_active_client():
    from data_science.sub_agents.bqml import tools as bqml

    with client_scope("npi"):
        train = bqml.get_revenue_training_sql()
        forecast = bqml.get_revenue_forecast_sql(30)
    assert f"`{PROJECT}.astrobot_ml_npi.revenue_forecast`" in train
    assert f"`{PROJECT}.astrobot_npi.performance`" in train
    assert "venetian" not in train and "thelook" not in train
    assert f"{PROJECT}.astrobot_ml_npi.revenue_forecast" in forecast

    with client_scope("venetian"):
        assert "astrobot_ml_venetian" in bqml.get_revenue_forecast_sql(7)


def test_demo_bqml_uses_thelook_and_demo_dataset():
    from data_science.sub_agents.bqml import tools as bqml

    train = bqml.get_revenue_training_sql()
    assert f"{PROJECT}.astrobot_ml.revenue_forecast" in train
    assert "bigquery-public-data.thelook_ecommerce.order_items" in train


def test_segmentation_not_available_for_clients():
    from data_science.sub_agents.bqml import tools as bqml

    with client_scope("npi"), pytest.raises(RuntimeError):
        bqml.get_customer_segmentation_training_sql()


# ------------------------------------------------------------
# Training approval
# ------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Forecast revenue for 30 days given our budget constraints",
    "Revenue forecast for the Trainers category",
    "Forecast revenue next 30 days",
    "How did the training campaign perform?",
])
def test_not_training_requests(text):
    from data_science.sub_agents.bqml.tools import training_requested
    from data_science.tools import _is_explicit_model_training_request

    assert not training_requested(text)
    assert not _is_explicit_model_training_request(text)


@pytest.mark.parametrize("text", [
    "Train the revenue model",
    "Please retrain the forecasting model",
    "Create a new forecasting model",
    "train a customer segmentation model",
])
def test_training_requests(text):
    from data_science.sub_agents.bqml.tools import training_requested

    assert training_requested(text)


def test_existing_model_needs_explicit_replace(monkeypatch):
    from data_science.sub_agents.bqml import tools as bqml

    monkeypatch.setattr(bqml, "_model_exists", lambda name: True)
    refusal = bqml._training_refusal("Train the revenue model", "revenue_forecast")
    assert json.loads(refusal)["status"] == "EXISTS"
    assert bqml._training_refusal("Retrain the revenue model", "revenue_forecast") is None


def test_train_tool_ignores_model_request_without_user_intent(monkeypatch):
    from data_science.sub_agents.bqml import tools as bqml

    monkeypatch.setattr(bqml, "_model_exists", lambda name: False)
    monkeypatch.setattr(bqml, "_execute_approved_sql",
                        lambda *a: pytest.fail("BigQuery must not be called"))
    ctx = SimpleNamespace(state={"_current_user_request": "Forecast revenue for 30 days"})
    with client_scope("npi"):
        assert json.loads(bqml.train_revenue_model(ctx))["status"] == "NOT_APPROVED"


# ------------------------------------------------------------
# Forecast routing and labels follow the client
# ------------------------------------------------------------

@pytest.mark.parametrize("text,forecast,evaluation", [
    ("Forecast conversions for the next 30 days", True, False),
    ("Generate a 30-day revenue forecast", True, False),
    ("Evaluate the forecasting model", False, True),
    ("How accurate is the forecast model? Show diagnostics", False, True),
    ("Explainable forecast for 14 days", False, False),
    ("Forecast customer segments", False, False),
])
def test_forecast_routing(text, forecast, evaluation):
    from data_science.tools import _is_revenue_evaluation_request, _is_revenue_forecast_request

    assert _is_revenue_forecast_request(text) is forecast
    assert _is_revenue_evaluation_request(text) is evaluation


def test_forecast_label_follows_client():
    from data_science.tools import _forecast_label

    assert _forecast_label() == "Revenue"
    with client_scope("npi"):
        assert _forecast_label() == "Conversions"
