"""Identity middleware against the real AstroBot ADK web app (IAP signature
check is stubbed; everything else is real)."""

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from astrobot import identity

ME = "analyst@example.com"
TOKEN = {"x-goog-iap-jwt-assertion": "signed.jwt.token"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "iap")
    monkeypatch.setenv("ASTROBOT_IAP_AUDIENCE", "/projects/1/locations/r/services/s")
    monkeypatch.setattr(identity, "_verify_iap_token",
                        lambda token: ME if token == "signed.jwt.token" else (_ for _ in ()).throw(identity.AuthError("bad")))
    from astrobot.web import app
    return TestClient(app)


def test_requests_without_sign_in_are_rejected(client):
    assert client.get("/list-apps").status_code == 401
    assert client.get("/apps/orchestrator/users/user/sessions").status_code == 401
    assert client.get("/list-apps", headers={"x-goog-iap-jwt-assertion": "forged"}).status_code == 401


def test_health_is_open(client):
    assert client.get("/health").status_code == 200


def test_sessions_are_created_and_listed_as_the_verified_user(client):
    created = client.post("/apps/orchestrator/users/someone-else/sessions", headers=TOKEN, json={})
    assert created.status_code == 200 and created.json()["userId"] == ME
    # Asking for "someone-else" still returns only the verified user's sessions.
    listed = client.get("/apps/orchestrator/users/someone-else/sessions", headers=TOKEN).json()
    assert [s["userId"] for s in listed] == [ME]


def test_other_users_session_cannot_be_read(client, monkeypatch):
    mine = client.post("/apps/orchestrator/users/x/sessions", headers=TOKEN, json={}).json()["id"]
    monkeypatch.setattr(identity, "_verify_iap_token", lambda token: "intruder@example.com")
    response = client.get(f"/apps/orchestrator/users/{ME}/sessions/{mine}", headers=TOKEN)
    assert response.status_code == 404


@pytest.mark.parametrize("path", [
    "/debug/trace/session/abc", "/debug/trace/evt", "/docs", "/openapi.json",
    "/apps/orchestrator/eval_sets", "/builder/app/orchestrator",
])
def test_developer_endpoints_blocked_in_production(client, path):
    assert client.get(path, headers=TOKEN).status_code == 404


def test_developer_endpoints_available_in_dev(monkeypatch):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "dev")
    from astrobot.web import app
    client = TestClient(app)
    assert client.get("/debug/trace/session/abc").json() != {"detail": "Not available."}
    assert client.get("/list-apps").json() == ["orchestrator"]


def test_unknown_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "off")
    assert identity.auth_mode() == "iap"


def test_run_body_user_is_replaced():
    captured = {}

    async def inner(scope, receive, send):
        message = await receive()
        captured["body"] = json.loads(message["body"])
        captured["length"] = dict(scope["headers"])[b"content-length"]

    async def go():
        body = json.dumps({"appName": "orchestrator", "userId": "victim@example.com",
                           "user_id": "x", "sessionId": "s"}).encode()
        messages = [{"type": "http.request", "body": body, "more_body": False}]
        scope = {"type": "http", "method": "POST", "path": "/run",
                 "headers": [(b"content-length", str(len(body)).encode())]}
        await identity.IdentityMiddleware(inner)(scope, lambda: _pop(messages), _noop)

    async def _pop(messages):
        return messages.pop(0)

    async def _noop(message):
        pass

    import os
    os.environ["ASTROBOT_AUTH_MODE"] = "dev"
    os.environ["ASTROBOT_DEV_USER"] = "dev-user"
    try:
        asyncio.run(go())
    finally:
        os.environ.pop("ASTROBOT_AUTH_MODE")
        os.environ.pop("ASTROBOT_DEV_USER")
    assert captured["body"]["userId"] == "dev-user" and "user_id" not in captured["body"]
    assert int(captured["length"]) == len(json.dumps(captured["body"]))


def test_token_checks(monkeypatch):
    from google.oauth2 import id_token

    monkeypatch.delenv("ASTROBOT_IAP_AUDIENCE", raising=False)
    with pytest.raises(identity.AuthError, match="not configured"):
        identity._verify_iap_token("t")

    monkeypatch.setenv("ASTROBOT_IAP_AUDIENCE", "/projects/1/locations/r/services/s")
    monkeypatch.setattr(id_token, "verify_token",
                        lambda *a, **k: {"iss": "https://evil.example", "email": ME})
    with pytest.raises(identity.AuthError, match="issuer"):
        identity._verify_iap_token("t")

    monkeypatch.setattr(id_token, "verify_token",
                        lambda *a, **k: {"iss": identity.IAP_ISSUER, "email": "Analyst@Example.com"})
    assert identity._verify_iap_token("t") == ME

    def boom(*a, **k):
        raise ValueError("bad signature")

    monkeypatch.setattr(id_token, "verify_token", boom)
    with pytest.raises(identity.AuthError, match="Invalid"):
        identity._verify_iap_token("t")


# ------------------------------------------------------------
# cloudrun mode (Cloud Run IAM auth; Google verified the token upstream)
# ------------------------------------------------------------

import base64 as _b64


def _token(**claims):
    body = _b64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"Bearer eyJhbGciOiJSUzI1NiJ9.{body}.SIGNATURE_REMOVED_BY_GOOGLE"


@pytest.fixture
def cloudrun(monkeypatch):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "cloudrun")
    from astrobot.web import app
    return TestClient(app)


def test_cloudrun_uses_verified_google_email(cloudrun):
    headers = {"authorization": _token(iss="https://accounts.google.com", email="Me@Example.com", email_verified=True)}
    created = cloudrun.post("/apps/orchestrator/users/user/sessions", headers=headers, json={})
    assert created.json()["userId"] == "me@example.com"


@pytest.mark.parametrize("claims", [
    {"iss": "https://evil.example", "email": "me@example.com", "email_verified": True},
    {"iss": "accounts.google.com", "email": "me@example.com", "email_verified": False},
    {"iss": "accounts.google.com", "email_verified": True},
])
def test_cloudrun_rejects_bad_tokens(cloudrun, claims):
    assert cloudrun.get("/list-apps", headers={"authorization": _token(**claims)}).status_code == 401


def test_cloudrun_rejects_missing_token(cloudrun):
    assert cloudrun.get("/list-apps").status_code == 401
    assert cloudrun.get("/list-apps", headers={"authorization": "Bearer not-a-jwt"}).status_code == 401


def test_cloudrun_blocks_developer_endpoints(cloudrun):
    headers = {"authorization": _token(iss="accounts.google.com", email="me@example.com", email_verified=True)}
    assert cloudrun.get("/debug/trace/session/x", headers=headers).status_code == 404


def test_cloudrun_reads_x_serverless_authorization(cloudrun):
    token = _token(iss="https://accounts.google.com", email="me@example.com", email_verified="true")
    created = cloudrun.post("/apps/orchestrator/users/user/sessions",
                            headers={"x-serverless-authorization": token}, json={})
    assert created.status_code == 200 and created.json()["userId"] == "me@example.com"


def test_cloudrun_missing_header_reason_names_headers_only(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        with pytest.raises(identity.AuthError) as err:
            import os
            os.environ["ASTROBOT_AUTH_MODE"] = "cloudrun"
            try:
                identity.resolve_user({b"x-goog-something": b"secret-value"})
            finally:
                os.environ.pop("ASTROBOT_AUTH_MODE")
    assert "x-goog-something" in str(err.value) and "secret-value" not in str(err.value)
