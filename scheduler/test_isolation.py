"""Scheduler isolation: client from the server scope, strict ownership,
named channels only, run-time access re-check, honest delivery status."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import scheduler.agent as agent
from scheduler import delivery, worker
from scheduler.store import ScheduleStore
from tenancy import registry
from tenancy.context import client_scope


@pytest.fixture
def store(tmp_path, monkeypatch):
    registry.reload()
    s = ScheduleStore(tmp_path / "schedules.json")
    monkeypatch.setattr(agent, "STORE", s)
    return s


def ctx(user="user", **state):
    return SimpleNamespace(state={"user_email": user, **state})


def _create(client, user="user", channels=None, **state):
    with client_scope(client):
        return agent.create_schedule("Weekly", "Weekly spend summary",
                                     cron="0 9 * * 1", channels=channels,
                                     tool_context=ctx(user, **state))


def test_client_comes_from_server_scope_not_state(store):
    payload = json.loads(_create("npi", client_id="venetian"))
    assert payload["schedule"]["context"]["client_id"] == "npi"


def test_no_client_no_schedule(store):
    assert _create("").startswith("Error")


def test_unknown_user_cannot_create(store):
    assert "signed-in user" in _create("npi", user="unknown")


def test_unsupported_channel_rejected(store):
    assert "not supported" in _create("npi", channels=["email"])


def test_google_chat_needs_client_configuration(store, monkeypatch):
    assert "not configured" in _create("npi", channels=["google_chat"])
    clients = {k: dict(v) for k, v in registry.clients().items()}
    clients["npi"]["google_chat"] = {"webhook_secret": "astrobot-gchat-npi"}
    monkeypatch.setattr(registry, "clients", lambda: clients)
    assert json.loads(_create("npi", channels=["google_chat"]))["status"] == "success"


def test_only_owner_can_delete(store):
    schedule_id = json.loads(_create("npi"))["schedule"]["id"]
    for user in ("someone-else", "unknown", ""):
        with client_scope("npi"):
            assert "Access denied" in agent.delete_schedule(schedule_id, tool_context=ctx(user))
    with client_scope("venetian"):
        assert "Access denied" in agent.delete_schedule(schedule_id, tool_context=ctx("user"))
    with client_scope("npi"):
        assert "Access denied" not in agent.delete_schedule(schedule_id, tool_context=ctx("user"))


def test_listing_shows_only_own_schedules(store):
    _create("npi", user="user")
    with client_scope("npi"):
        theirs = json.loads(agent.list_schedules(tool_context=ctx("other-analyst")))
        mine = json.loads(agent.list_schedules(tool_context=ctx("user")))
    assert theirs["count"] == 0 and mine["count"] == 1


# ------------------------------------------------------------
# Worker: access re-checked at run time
# ------------------------------------------------------------

def test_run_denied_when_owner_lost_access(store, monkeypatch):
    schedule_id = json.loads(_create("npi"))["schedule"]["id"]
    monkeypatch.setattr(registry, "has_access", lambda user, client: False)
    monkeypatch.setattr(worker, "Runner", lambda **k: pytest.fail("must not run"))
    result = asyncio.run(worker.execute_schedule(schedule_id, force=True, store=store))
    assert result["status"] == "access_denied"


# ------------------------------------------------------------
# Delivery
# ------------------------------------------------------------

def test_internal_and_unconfigured_chat_report_honestly():
    registry.reload()
    results = delivery.deliver("npi", ["internal", "google_chat"], "t", "x")
    assert results == [{"channel": "internal", "status": "stored"},
                       {"channel": "google_chat", "status": "not_configured"}]


def _configure_chat(monkeypatch, url="https://chat.googleapis.com/v1/spaces/AAA/messages?key=k"):
    monkeypatch.setattr(delivery, "_chat_secret_name", lambda c: "astrobot-gchat-npi")
    monkeypatch.setattr(delivery, "_webhook_url", lambda name: url)


def test_chat_delivery_posts_and_truncates(monkeypatch):
    _configure_chat(monkeypatch)
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, text=json["text"])
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(delivery.requests, "post", post)
    result = delivery.deliver("npi", ["google_chat"], "Weekly", "x" * 9000)
    assert result == [{"channel": "google_chat", "status": "delivered"}]
    assert len(sent["text"]) <= delivery.CHAT_TEXT_LIMIT and "truncated" in sent["text"]


def test_chat_failure_is_reported_without_url(monkeypatch, caplog):
    _configure_chat(monkeypatch)

    def post(*a, **k):
        raise delivery.requests.ConnectionError("https://chat.googleapis.com/secret")

    monkeypatch.setattr(delivery.requests, "post", post)
    result = delivery.deliver("npi", ["google_chat"], "t", "x")
    assert result[0]["status"] == "failed"
    assert "chat.googleapis.com" not in json.dumps(result) + caplog.text


def test_non_chat_webhook_secret_is_rejected(monkeypatch):
    monkeypatch.setattr(delivery, "_chat_secret_name", lambda c: "s")

    class FakeSM:
        def access_secret_version(self, request):
            return SimpleNamespace(payload=SimpleNamespace(data=b"https://evil.example.com/hook"))

    import google.cloud.secretmanager as sm
    monkeypatch.setattr(sm, "SecretManagerServiceClient", lambda: FakeSM())
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setattr(delivery.requests, "post", lambda *a, **k: pytest.fail("must not post"))
    assert delivery.deliver("npi", ["google_chat"], "t", "x")[0]["status"] == "failed"


def test_update_channels_validated(store):
    schedule_id = json.loads(_create("npi"))["schedule"]["id"]
    with client_scope("npi"):
        assert "not supported" in agent.update_schedule(schedule_id, channels=["email"], tool_context=ctx())
        assert "not configured" in agent.update_schedule(schedule_id, channels=["google_chat"], tool_context=ctx())
        ok = json.loads(agent.update_schedule(schedule_id, channels=["internal"], tool_context=ctx()))
    assert ok["schedule"]["channels"] == ["internal"]
