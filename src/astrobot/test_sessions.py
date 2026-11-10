"""Durable sessions: the astrobot-agentengine:// URI builds the owner-checked
service, and delete refuses sessions owned by someone else."""

import asyncio

import pytest
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

from astrobot import sessions


def test_uri_builds_owner_checked_service():
    sessions.register()
    from google.adk.cli.service_registry import get_service_registry

    service = get_service_registry().create_session_service(
        "astrobot-agentengine://projects/p/locations/us-central1/reasoningEngines/123"
    )
    assert isinstance(service, sessions.OwnerCheckedVertexAiSessionService)


def _service(monkeypatch, owner):
    deleted = []

    async def fake_get(self, *, app_name, user_id, session_id, config=None):
        if owner is None:
            return None
        if owner != user_id:
            raise ValueError(f"Session {session_id} does not belong to user {user_id}.")
        return object()

    async def fake_delete(self, *, app_name, user_id, session_id):
        deleted.append(session_id)

    monkeypatch.setattr(VertexAiSessionService, "get_session", fake_get)
    monkeypatch.setattr(VertexAiSessionService, "delete_session", fake_delete)
    svc = sessions.OwnerCheckedVertexAiSessionService(
        project="p", location="us-central1", agent_engine_id="123")
    return svc, deleted


def test_owner_can_delete(monkeypatch):
    svc, deleted = _service(monkeypatch, owner="me@x.com")
    asyncio.run(svc.delete_session(app_name="a", user_id="me@x.com", session_id="s1"))
    assert deleted == ["s1"]


def test_other_user_cannot_delete(monkeypatch):
    svc, deleted = _service(monkeypatch, owner="me@x.com")
    with pytest.raises(ValueError):
        asyncio.run(svc.delete_session(app_name="a", user_id="intruder@x.com", session_id="s1"))
    assert deleted == []


def test_missing_session_is_a_no_op(monkeypatch):
    svc, deleted = _service(monkeypatch, owner=None)
    asyncio.run(svc.delete_session(app_name="a", user_id="me@x.com", session_id="gone"))
    assert deleted == []
