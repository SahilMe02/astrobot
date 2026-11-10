"""Public-URL mode: Google sign-in done by the app itself (Google's token
endpoint and signature check are stubbed; everything else is real)."""

from urllib.parse import parse_qs, urlparse

import pytest
from starlette.testclient import TestClient

from astrobot import google_login

APPROVED = "astrobot.test012@gmail.com"  # in tenancy/access.json


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "google")
    monkeypatch.setenv("ASTROBOT_PUBLIC_URL", "https://astrobot.example.run.app")
    monkeypatch.setenv("ASTROBOT_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com")
    monkeypatch.setenv("ASTROBOT_OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("ASTROBOT_SESSION_KEY", "k" * 40)
    monkeypatch.setattr(google_login, "_exchange_code", lambda code: f"idtoken-for-{code}")
    from astrobot.web import app
    return TestClient(app, base_url="https://astrobot.example.run.app")


def _claims(monkeypatch, **claims):
    monkeypatch.setattr(google_login, "_verify_id_token", lambda token: claims)


def _sign_in(client, monkeypatch, email=APPROVED, verified=True):
    _claims(monkeypatch, email=email, email_verified=verified)
    login = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    return client.get(f"/auth/callback?code=c&state={state}", follow_redirects=False)


def test_browser_without_session_is_sent_to_google(client):
    page = client.get("/dev-ui/", headers={"accept": "text/html"}, follow_redirects=False)
    assert page.status_code == 302 and page.headers["location"] == "/auth/login"
    login = client.get("/auth/login", follow_redirects=False)
    url = urlparse(login.headers["location"])
    params = parse_qs(url.query)
    assert url.netloc == "accounts.google.com"
    assert params["redirect_uri"] == ["https://astrobot.example.run.app/auth/callback"]
    assert params["scope"] == ["openid email"]


def test_api_without_session_is_401(client):
    assert client.get("/list-apps").status_code == 401
    assert client.post("/apps/orchestrator/users/x/sessions", json={}).status_code == 401


def test_approved_user_signs_in_and_runs_as_themselves(client, monkeypatch):
    done = _sign_in(client, monkeypatch)
    assert done.status_code == 302 and done.headers["location"] == "/dev-ui/"
    cookie = done.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=Lax" in cookie
    created = client.post("/apps/orchestrator/users/someone-else/sessions", json={})
    assert created.status_code == 200 and created.json()["userId"] == APPROVED


def test_unapproved_or_unverified_user_gets_no_session(client, monkeypatch):
    refused = _sign_in(client, monkeypatch, email="stranger@gmail.com")
    assert refused.status_code == 403 and "not approved" in refused.text
    assert client.get("/list-apps").status_code == 401
    assert _sign_in(client, monkeypatch, verified=False).status_code == 403


def test_callback_needs_matching_state(client, monkeypatch):
    _claims(monkeypatch, email=APPROVED, email_verified=True)
    client.get("/auth/login", follow_redirects=False)
    assert client.get("/auth/callback?code=c&state=forged", follow_redirects=False).status_code == 403


def test_forged_or_tampered_cookie_is_rejected(client, monkeypatch):
    other = TestClient(client.app, base_url="https://astrobot.example.run.app")
    other.cookies.set("astrobot_session", "eyJlbWFpbCI6ImFAYi5jIn0.bad")
    assert other.get("/list-apps").status_code == 401
    # A cookie signed with another key is useless.
    monkeypatch.setenv("ASTROBOT_SESSION_KEY", "z" * 40)
    forged = google_login.sign({"email": APPROVED}, 60)
    monkeypatch.setenv("ASTROBOT_SESSION_KEY", "k" * 40)
    other.cookies.set("astrobot_session", forged)
    assert other.get("/list-apps").status_code == 401


def test_cloud_run_style_fake_token_does_not_work_in_public_mode(client):
    import base64, json
    body = base64.urlsafe_b64encode(json.dumps(
        {"iss": "https://accounts.google.com", "email": APPROVED, "email_verified": True}
    ).encode()).decode().rstrip("=")
    fake = {"authorization": f"Bearer x.{body}.y", "x-serverless-authorization": f"Bearer x.{body}.y"}
    assert client.get("/list-apps", headers=fake).status_code == 401


def test_expired_session_and_logout(client, monkeypatch):
    _sign_in(client, monkeypatch)
    assert client.get("/list-apps").status_code == 200
    client.get("/auth/logout")
    assert client.get("/list-apps").status_code == 401


def test_revoked_access_ends_session(client, monkeypatch):
    _sign_in(client, monkeypatch)
    from tenancy import registry
    monkeypatch.setattr(registry, "allowed_clients", lambda user: frozenset())
    assert client.get("/list-apps").status_code == 401


def test_developer_endpoints_still_blocked(client, monkeypatch):
    _sign_in(client, monkeypatch)
    assert client.get("/debug/trace/session/x").status_code == 404
    assert client.get("/docs").status_code == 404


def test_missing_configuration_fails_closed(client, monkeypatch):
    monkeypatch.delenv("ASTROBOT_OAUTH_CLIENT_ID")
    assert client.get("/auth/login", follow_redirects=False).status_code == 403
