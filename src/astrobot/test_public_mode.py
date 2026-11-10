"""Public demo link: no sign-in, but every visitor is a separate, signed
anonymous user, and guests only get the clients listed under "guests"."""

import pytest
from starlette.testclient import TestClient

from astrobot import google_login, identity
from tenancy import registry

BASE = "https://astrobot.example.run.app"


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "public")
    monkeypatch.setenv("ASTROBOT_PUBLIC_URL", BASE)
    monkeypatch.setenv("ASTROBOT_SESSION_KEY", "k" * 40)
    registry.reload()
    from astrobot.web import app
    return lambda: TestClient(app, base_url=BASE)


def _visit(client):
    first = client.get("/dev-ui/", headers={"accept": "text/html"}, follow_redirects=False)
    assert first.status_code == 302 and first.headers["location"] == "/dev-ui/"
    cookie = first.headers["set-cookie"]
    assert cookie.startswith("astrobot_guest=") and "HttpOnly" in cookie and "Secure" in cookie
    return client


def test_visitor_gets_own_id_and_can_use_app(make_client):
    client = _visit(make_client())
    created = client.post("/apps/orchestrator/users/user/sessions", json={})
    assert created.status_code == 200
    assert identity._GUEST_ID.fullmatch(created.json()["userId"])


def test_two_visitors_never_see_each_others_sessions(make_client):
    alice, bob = _visit(make_client()), _visit(make_client())
    session = alice.post("/apps/orchestrator/users/x/sessions", json={}).json()
    owner = session["userId"]
    assert bob.get("/apps/orchestrator/users/x/sessions").json() == []
    assert bob.get(f"/apps/orchestrator/users/{owner}/sessions/{session['id']}").status_code == 404


def test_api_without_cookie_is_401(make_client):
    assert make_client().post("/apps/orchestrator/users/x/sessions", json={}).status_code == 401


def test_forged_cookie_and_fake_tokens_rejected(make_client, monkeypatch):
    client = make_client()
    client.cookies.set("astrobot_guest", "eyJndWVzdCI6Imd1ZXN0LTAwMDAwMDAwMDAwMDAwMDAifQ.bad")
    assert client.get("/list-apps").status_code == 401
    # A real email can't be smuggled in via a cookie signed with another key...
    monkeypatch.setenv("ASTROBOT_SESSION_KEY", "z" * 40)
    forged = google_login.sign({"guest": "guest-0000000000000000"}, 60)
    monkeypatch.setenv("ASTROBOT_SESSION_KEY", "k" * 40)
    client.cookies.set("astrobot_guest", forged)
    assert client.get("/list-apps").status_code == 401
    # ...nor as a correctly signed non-guest ID.
    client.cookies.set("astrobot_guest", google_login.sign({"guest": "sahilpmehta02@gmail.com"}, 60))
    assert client.get("/list-apps").status_code == 401


def test_developer_endpoints_blocked(make_client):
    client = _visit(make_client())
    assert client.get("/debug/trace/session/x").status_code == 404
    assert client.get("/docs").status_code == 404


def test_no_open_redirect(make_client):
    response = make_client().get("//evil.example/x", headers={"accept": "text/html"},
                                 follow_redirects=False)
    assert not response.headers.get("location", "/").startswith("//")


def test_guest_access_only_in_public_mode(monkeypatch):
    registry.reload()
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "public")
    assert registry.allowed_clients("guest-0123456789abcdef") == {"npi", "venetian", "winndixie"}
    for mode in ("cloudrun", "google", "iap", "dev"):
        monkeypatch.setenv("ASTROBOT_AUTH_MODE", mode)
        assert registry.allowed_clients("guest-0123456789abcdef") == frozenset()


def test_missing_key_fails_closed(monkeypatch):
    monkeypatch.setenv("ASTROBOT_AUTH_MODE", "public")
    monkeypatch.setenv("ASTROBOT_PUBLIC_URL", BASE)
    monkeypatch.delenv("ASTROBOT_SESSION_KEY", raising=False)
    from astrobot.web import app
    client = TestClient(app, base_url=BASE)
    assert client.get("/dev-ui/", headers={"accept": "text/html"}, follow_redirects=False).status_code == 500
    assert client.get("/list-apps").status_code == 401
