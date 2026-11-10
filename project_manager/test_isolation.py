"""Project Manager isolation: mailbox scope comes from the registry, model
terms can only narrow it, and only this client's threads can be opened."""

from types import SimpleNamespace

import pytest

from project_manager.tools import gmail_tool as gt
from tenancy import registry
from tenancy.context import client_scope


@pytest.fixture
def gmail(monkeypatch):
    registry.reload()
    calls = []

    def fake_get(session, path, params=None):
        calls.append((path, dict(params or {})))
        if path == "threads":
            return {"threads": [{"id": "t-venetian-1"}, {"id": "t-venetian-2"}]}
        if path.startswith("threads/"):
            return {"id": path.split("/")[1], "messages": []}
        if path.startswith("messages/"):
            owner = "t-venetian-1" if path.endswith("own") else "t-npi-9"
            return {"id": path.split("/")[1], "threadId": owner, "payload": {}}
        return {}

    monkeypatch.setattr(gt, "_configuration_error", lambda: None)
    monkeypatch.setattr(gt, "_get_gmail_session", lambda: object())
    monkeypatch.setattr(gt, "_gmail_get", fake_get)
    return calls


def ctx():
    return SimpleNamespace(state={})


def test_search_uses_registry_filter_for_active_client(gmail):
    with client_scope("venetian"):
        gt.fetch_recent_threads(client_id="venetian", tool_context=ctx())
    assert gmail[-1][1]["q"] == '("venetian")'


def test_model_cannot_pick_another_client_via_argument(gmail):
    with client_scope("venetian"):
        result = gt.fetch_recent_threads(client_id="npi", tool_context=ctx())
    assert "Access denied" in result["error"] and gmail == []


def test_extra_terms_only_narrow_the_scope(gmail):
    with client_scope("venetian"):
        gt.fetch_recent_threads(
            client_id="", query='OR "npi") OR (label:inbox', tool_context=ctx()
        )
    q = gmail[-1][1]["q"]
    assert q.startswith('("venetian") (') and q.count("(") == 2 and q.count(")") == 2


def test_no_client_no_mail(gmail):
    result = gt.fetch_recent_threads(client_id="venetian", tool_context=ctx())
    assert "client context is required" in result["error"] and gmail == []


def test_client_without_mailbox_config_fails_closed(gmail, monkeypatch):
    clients = {k: dict(v) for k, v in registry.clients().items()}
    clients["venetian"].pop("gmail")
    monkeypatch.setattr(registry, "clients", lambda: clients)
    with client_scope("venetian"):
        result = gt.fetch_recent_threads(client_id="venetian", tool_context=ctx())
    assert "not configured" in result["error"]
    assert all(path != "threads" for path, _ in gmail)


def test_only_threads_from_scoped_search_can_be_opened(gmail):
    c = ctx()
    with client_scope("venetian"):
        denied = gt.get_thread_messages_full("t-npi-9", "venetian", tool_context=c)
        assert "Access denied" in denied["error"]
        gt.fetch_recent_threads(client_id="venetian", tool_context=c)
        opened = gt.get_thread_messages_full("t-venetian-1", "venetian", tool_context=c)
    assert "error" not in opened
    assert not any(path == "threads/t-npi-9" for path, _ in gmail)


def test_thread_allow_list_is_per_client(gmail):
    c = ctx()
    with client_scope("venetian"):
        gt.fetch_recent_threads(client_id="venetian", tool_context=c)
    with client_scope("npi"):
        result = gt.extract_thread_snippets("t-venetian-1", "npi", tool_context=c)
    assert "Access denied" in result["error"]


def test_attachments_only_from_allowed_threads(gmail):
    c = ctx()
    with client_scope("venetian"):
        gt.fetch_recent_threads(client_id="venetian", tool_context=c)
        foreign = gt.get_message_attachments("m-foreign", "venetian", tool_context=c)
        own = gt.get_message_attachments("m-own", "venetian", tool_context=c)
    assert "Access denied" in foreign["error"]
    assert "error" not in own


def test_email_evidence_fetched_once_per_request(monkeypatch):
    from project_manager import agent as pm

    fetches = []
    monkeypatch.setattr(pm, "collect_recent_project_email_context",
                        lambda **k: fetches.append(1) or {"threads": []})
    state = {"_current_user_request": "Any blockers in recent emails?"}
    cb = SimpleNamespace(state=state, invocation_id="inv-1")
    request = SimpleNamespace(config=SimpleNamespace(system_instruction=None))
    with client_scope("venetian"):
        pm.before_model_callback(cb, request)
        pm.before_model_callback(cb, request)
        pm.before_model_callback(SimpleNamespace(state=state, invocation_id="inv-2"), request)
    assert len(fetches) == 2
