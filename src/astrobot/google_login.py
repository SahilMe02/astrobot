""""Sign in with Google" for the public AstroBot URL (ASTROBOT_AUTH_MODE=google).

The service is public, so nothing upstream checks the caller. This module
does the whole check itself:

  /auth/login     redirect to Google (scope: openid email)
  /auth/callback  exchange the code, VERIFY the ID token signature, audience
                  and expiry, require a verified email that is on the access
                  list (tenancy/access.json), then set a signed session cookie
  /auth/logout    clear the cookie

Every other request must carry a valid, unexpired, HMAC-signed session
cookie. People not on the access list never get one.

Settings (env; secrets via Cloud Run --set-secrets):
  ASTROBOT_PUBLIC_URL            e.g. https://astrobot-web-xxxx.us-central1.run.app
  ASTROBOT_OAUTH_CLIENT_ID       OAuth "Web application" client ID
  ASTROBOT_OAUTH_CLIENT_SECRET   its secret
  ASTROBOT_SESSION_KEY           random key that signs session cookies
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from http.cookies import SimpleCookie
from urllib.parse import urlencode

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SESSION_COOKIE = "astrobot_session"
STATE_COOKIE = "astrobot_oauth_state"
SESSION_SECONDS = 8 * 3600
STATE_SECONDS = 600


class LoginError(Exception):
    pass


def _setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LoginError(f"{name} is not configured.")
    return value


def _public_url() -> str:
    return _setting("ASTROBOT_PUBLIC_URL").rstrip("/")


def _redirect_uri() -> str:
    return _public_url() + "/auth/callback"


def _key() -> bytes:
    key = _setting("ASTROBOT_SESSION_KEY")
    if len(key) < 32:
        raise LoginError("ASTROBOT_SESSION_KEY must be at least 32 characters.")
    return key.encode()


# ------------------------------------------------------------
# Signed values
# ------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def sign(payload: dict, ttl: int) -> str:
    body = _b64(json.dumps({**payload, "exp": int(time.time()) + ttl}).encode())
    mac = _b64(hmac.new(_key(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{mac}"


def unsign(value: str) -> dict | None:
    body, _, mac = (value or "").partition(".")
    if not body or not mac:
        return None
    expected = _b64(hmac.new(_key(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(mac, expected):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("exp", 0) < time.time():
        return None
    return data


# ------------------------------------------------------------
# Cookies
# ------------------------------------------------------------

def read_cookie(headers: dict[bytes, bytes], name: str) -> str:
    jar = SimpleCookie()
    try:
        jar.load(headers.get(b"cookie", b"").decode("latin-1"))
    except Exception:
        return ""
    return jar[name].value if name in jar else ""


def set_cookie(name: str, value: str, max_age: int) -> str:
    secure = "; Secure" if _public_url().startswith("https://") else ""
    return f"{name}={value}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Lax{secure}"


# ------------------------------------------------------------
# Flow
# ------------------------------------------------------------

def session_user(headers: dict[bytes, bytes]) -> str:
    """Email from a valid session cookie, or raise LoginError."""
    data = unsign(read_cookie(headers, SESSION_COOKIE))
    email = str((data or {}).get("email", "")).strip().lower()
    if not email:
        raise LoginError("Sign-in required.")
    from tenancy import registry

    if not registry.allowed_clients(email):  # access removed since sign-in
        raise LoginError("Access revoked.")
    return email


def start_login() -> tuple[str, str]:
    """(Google URL to redirect to, Set-Cookie header holding the state)."""
    state = secrets.token_urlsafe(24)
    params = {
        "client_id": _setting("ASTROBOT_OAUTH_CLIENT_ID"),
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": "openid email",
        "state": state,
        "prompt": "select_account",
    }
    cookie = set_cookie(STATE_COOKIE, sign({"state": state}, STATE_SECONDS), STATE_SECONDS)
    return f"{AUTH_URL}?{urlencode(params)}", cookie


def _exchange_code(code: str) -> str:
    import requests

    response = requests.post(TOKEN_URL, timeout=15, data={
        "code": code,
        "client_id": _setting("ASTROBOT_OAUTH_CLIENT_ID"),
        "client_secret": _setting("ASTROBOT_OAUTH_CLIENT_SECRET"),
        "redirect_uri": _redirect_uri(),
        "grant_type": "authorization_code",
    })
    if response.status_code != 200:
        raise LoginError("Google sign-in failed.")
    token = response.json().get("id_token")
    if not token:
        raise LoginError("Google returned no identity token.")
    return token


def _verify_id_token(token: str) -> dict:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    try:
        # Checks signature, audience (our client ID), expiry and issuer.
        return id_token.verify_oauth2_token(
            token, google_requests.Request(), _setting("ASTROBOT_OAUTH_CLIENT_ID")
        )
    except Exception as exc:
        raise LoginError("Invalid Google identity token.") from exc


def finish_login(query: dict[str, list[str]], headers: dict[bytes, bytes]) -> str:
    """Validate the callback and return the approved user's email."""
    state = (query.get("state") or [""])[0]
    saved = unsign(read_cookie(headers, STATE_COOKIE)) or {}
    if not state or not hmac.compare_digest(state, str(saved.get("state", ""))):
        raise LoginError("Sign-in expired. Please try again.")
    code = (query.get("code") or [""])[0]
    if not code:
        raise LoginError("Sign-in was cancelled.")

    claims = _verify_id_token(_exchange_code(code))
    if claims.get("email_verified") not in (True, "true"):
        raise LoginError("Your Google email is not verified.")
    email = str(claims.get("email", "")).strip().lower()

    from tenancy import registry

    if not email or not registry.allowed_clients(email):
        raise LoginError(f"{email or 'This account'} is not approved for AstroBot.")
    return email
