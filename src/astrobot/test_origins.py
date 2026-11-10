"""ADK's cross-site check: the proxy origin must be explicitly allowed;
anything else (and "*") stays blocked."""

import importlib

from starlette.testclient import TestClient


def _client(monkeypatch, origins):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "dev")
    monkeypatch.setenv("ASTROBOT_ALLOWED_ORIGINS", origins)
    import astrobot.web as web
    return TestClient(importlib.reload(web).app, base_url="https://astrobot-web.example.run.app")


def test_proxy_origin_allowed(monkeypatch):
    client = _client(monkeypatch, "http://127.0.0.1:8081,http://localhost:8081")
    r = client.post("/apps/orchestrator/users/user/sessions", json={},
                    headers={"origin": "http://127.0.0.1:8081"})
    assert r.status_code == 200


def test_other_origin_blocked(monkeypatch):
    client = _client(monkeypatch, "http://127.0.0.1:8081")
    r = client.post("/apps/orchestrator/users/user/sessions", json={},
                    headers={"origin": "https://evil.example.com"})
    assert r.status_code == 403


def test_wildcard_is_ignored(monkeypatch):
    client = _client(monkeypatch, "*")
    r = client.post("/apps/orchestrator/users/user/sessions", json={},
                    headers={"origin": "https://evil.example.com"})
    assert r.status_code == 403
