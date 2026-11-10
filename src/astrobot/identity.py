"""Verified user identity for the AstroBot web app.

ASGI middleware in front of the ADK FastAPI app.

ASTROBOT_AUTH_MODE:
  iap (default) - Cloud Run behind Identity-Aware Proxy. Every request must
                  carry a valid, signed IAP JWT; the user is its email claim.
  cloudrun      - Cloud Run with IAM authentication (--no-allow-unauthenticated).
                  Google verifies the caller's ID token and roles/run.invoker
                  BEFORE the request reaches the container, then strips the
                  token signature. The app reads the email from that already-
                  verified token. Only valid when the service is not public.
  google        - PUBLIC Cloud Run URL. The app runs "Sign in with Google"
                  itself and verifies the ID token signature; only emails on
                  the access list get a signed session cookie (google_login.py).
  public        - PUBLIC demo link, no sign-in. Each browser gets a random,
                  HMAC-signed visitor ID cookie ("guest-<16 hex>") so visitors
                  never see each other's sessions. Guests may use only the
                  "guests" clients in access.json.
  dev           - local development only; the user is ASTROBOT_DEV_USER.

In both modes the user ID in every session/artifact URL and in the /run
request body is replaced with the verified user, so a caller can never read
or write another user's sessions. In iap mode, developer endpoints that do
not check the user (traces, evals, builder, API docs, live websocket) are
blocked.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable
from html import escape
from urllib.parse import parse_qs, quote

from astrobot import google_login

logger = logging.getLogger("astrobot-identity")

IAP_CERTS_URL = "https://www.gstatic.com/iap/verify/public_key-jwk"
IAP_ISSUER = "https://cloud.google.com/iap"
IAP_HEADER = b"x-goog-iap-jwt-assertion"

_USER_PATH = re.compile(r"^(/apps/[^/]+/users/)([^/]+)(/.*)?$")
_RUN_PATHS = {"/run", "/run_sse"}
_BLOCKED_IN_PRODUCTION = re.compile(
    r"^/(debug/|builder/|docs|redoc|openapi\.json|run_live)"
    r"|^/apps/[^/]+/(eval|metrics-info)"
    r"|^/apps/[^/]+/users/[^/]+/memory"
)


class AuthError(Exception):
    pass


def auth_mode() -> str:
    mode = os.getenv("ASTROBOT_AUTH_MODE", "iap").strip().lower()
    return mode if mode in {"iap", "cloudrun", "google", "public", "dev"} else "iap"


GUEST_COOKIE = "astrobot_guest"
GUEST_SECONDS = 30 * 24 * 3600
_GUEST_ID = re.compile(r"guest-[0-9a-f]{16}")


def _guest_user(headers: dict[bytes, bytes]) -> str:
    """Anonymous visitor ID from a signed cookie the app issued."""
    data = google_login.unsign(google_login.read_cookie(headers, GUEST_COOKIE)) or {}
    guest = str(data.get("guest", ""))
    if not _GUEST_ID.fullmatch(guest):
        raise AuthError("No visitor cookie.")
    return guest


def new_guest_cookie() -> str:
    import secrets

    guest = f"guest-{secrets.token_hex(8)}"
    return google_login.set_cookie(
        GUEST_COOKIE, google_login.sign({"guest": guest}, GUEST_SECONDS), GUEST_SECONDS
    )


_GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}


def _cloud_run_caller(authorization: str) -> str:
    """Email from the Google ID token Cloud Run already verified."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token.count(".") != 2:
        raise AuthError("Sign-in required.")
    import base64

    payload = token.split(".")[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception as exc:
        raise AuthError("Malformed identity token.") from exc
    if claims.get("iss") not in _GOOGLE_ISSUERS:
        raise AuthError("Identity token not issued by Google.")
    if claims.get("email_verified") not in (True, "true"):
        raise AuthError("Email not verified.")
    email = str(claims.get("email", "")).strip().lower()
    if not email:
        raise AuthError("Identity token has no email.")
    return email


def _verify_iap_token(token: str) -> str:
    """Return the verified email from an IAP JWT, or raise AuthError."""
    audience = os.getenv("ASTROBOT_IAP_AUDIENCE", "").strip()
    if not audience:
        raise AuthError("ASTROBOT_IAP_AUDIENCE is not configured.")
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    try:
        claims = id_token.verify_token(
            token,
            google_requests.Request(),
            audience=audience,
            certs_url=IAP_CERTS_URL,
        )
    except Exception as exc:
        raise AuthError(f"Invalid IAP token: {type(exc).__name__}") from exc
    if claims.get("iss") != IAP_ISSUER:
        raise AuthError("Invalid IAP token issuer.")
    email = str(claims.get("email", "")).strip().lower()
    if not email:
        raise AuthError("IAP token has no email.")
    return email


def resolve_user(headers: dict[bytes, bytes]) -> str:
    mode = auth_mode()
    if mode == "dev":
        return os.getenv("ASTROBOT_DEV_USER", "user").strip().lower() or "user"
    if mode == "cloudrun":
        # The Cloud Run proxy and other Google clients may send the token in
        # X-Serverless-Authorization (validated by Cloud Run) instead of
        # Authorization; prefer it when present.
        raw = headers.get(b"x-serverless-authorization") or headers.get(b"authorization") or b""
        if not raw:
            present = sorted(
                k.decode("latin-1") for k in headers
                if b"auth" in k or k.startswith(b"x-goog") or k.startswith(b"x-serverless")
            )
            raise AuthError(f"No identity token header (auth-related headers present: {present or 'none'}).")
        return _cloud_run_caller(raw.decode("latin-1").strip())
    if mode == "public":
        try:
            return _guest_user(headers)
        except google_login.LoginError as exc:
            raise AuthError(str(exc)) from exc
    if mode == "google":
        try:
            return google_login.session_user(headers)
        except google_login.LoginError as exc:
            raise AuthError(str(exc)) from exc
    token = headers.get(IAP_HEADER, b"").decode("latin-1").strip()
    if not token:
        raise AuthError("Sign-in required.")
    return _verify_iap_token(token)


def rewrite_user_path(path: str, user: str) -> str:
    match = _USER_PATH.match(path)
    if not match:
        return path
    return f"{match.group(1)}{quote(user, safe='@.')}{match.group(3) or ''}"


def rewrite_run_body(body: bytes, user: str) -> bytes:
    data = json.loads(body or b"{}")
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    data.pop("user_id", None)
    data["userId"] = user
    return json.dumps(data).encode("utf-8")


class IdentityMiddleware:
    def __init__(self, app: Callable):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path == "/health":
            return await self.app(scope, receive, send)

        mode = auth_mode()
        if mode != "dev" and _BLOCKED_IN_PRODUCTION.match(path):
            return await _reject(scope, send, 404, "Not available.")

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        if mode == "google" and path.startswith("/auth/") and scope["type"] == "http":
            return await _google_auth_route(scope, send, path, headers)

        try:
            user = resolve_user(headers)
        except AuthError as exc:
            logger.warning("Rejected request | path=%s | %s", path, exc)
            if (mode == "public" and scope["type"] == "http"
                    and scope.get("method") == "GET"
                    and b"text/html" in headers.get(b"accept", b"")):
                # First visit: issue a private visitor ID, then reload.
                try:
                    cookie = new_guest_cookie()
                except google_login.LoginError:
                    logger.exception("Public mode is misconfigured")
                    return await _reject(scope, send, 500, "Service is not configured.")
                query = scope.get("query_string", b"").decode("latin-1")
                target = path if path.startswith("/") and not path.startswith("//") else "/"
                return await _redirect(send, target + (f"?{query}" if query else ""), [cookie])
            if (mode == "google" and scope["type"] == "http"
                    and scope.get("method") == "GET"
                    and b"text/html" in headers.get(b"accept", b"")):
                return await _redirect(send, "/auth/login")
            return await _reject(scope, send, 401, "Sign-in required.")

        scope = dict(scope)
        new_path = rewrite_user_path(path, user)
        if new_path != path:
            scope["path"] = new_path
            scope["raw_path"] = new_path.encode("utf-8")
        scope.setdefault("state", {})["astrobot_user"] = user

        if scope["type"] == "http" and scope.get("method") == "POST" and path in _RUN_PATHS:
            body = b""
            more = True
            while more:
                message = await receive()
                body += message.get("body", b"")
                more = message.get("more_body", False)
            try:
                body = rewrite_run_body(body, user)
            except ValueError:
                return await _reject(scope, send, 400, "Invalid request body.")
            scope["headers"] = [
                (k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"
            ] + [(b"content-length", str(len(body)).encode())]
            sent = False

            async def receive_rewritten():
                nonlocal sent
                if sent:
                    return await receive()
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}

            return await self.app(scope, receive_rewritten, send)

        return await self.app(scope, receive, send)


async def _google_auth_route(scope, send, path, headers):
    try:
        if path == "/auth/login":
            location, cookie = google_login.start_login()
            return await _redirect(send, location, [cookie])
        if path == "/auth/callback":
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            email = google_login.finish_login(query, headers)
            logger.info("Signed in | user=%s", email)
            session = google_login.set_cookie(
                google_login.SESSION_COOKIE,
                google_login.sign({"email": email}, google_login.SESSION_SECONDS),
                google_login.SESSION_SECONDS,
            )
            clear_state = google_login.set_cookie(google_login.STATE_COOKIE, "", 0)
            return await _redirect(send, "/dev-ui/", [session, clear_state])
        if path == "/auth/logout":
            cleared = google_login.set_cookie(google_login.SESSION_COOKIE, "", 0)
            return await _page(send, 200, "You are signed out.", [cleared])
    except google_login.LoginError as exc:
        logger.warning("Sign-in refused | %s", exc)
        return await _page(send, 403, str(exc))
    except Exception:
        logger.exception("Sign-in failed")
        return await _page(send, 500, "Sign-in failed. Please try again.")
    return await _page(send, 404, "Not found.")


async def _redirect(send, location: str, cookies: list[str] | None = None):
    headers = [(b"location", location.encode()), (b"content-length", b"0"),
               (b"cache-control", b"no-store")]
    headers += [(b"set-cookie", c.encode()) for c in cookies or []]
    await send({"type": "http.response.start", "status": 302, "headers": headers})
    await send({"type": "http.response.body", "body": b""})


async def _page(send, status: int, message: str, cookies: list[str] | None = None):
    body = (
        "<!doctype html><meta charset=utf-8><title>AstroBot</title>"
        "<body style='font-family:system-ui;max-width:32rem;margin:4rem auto'>"
        f"<h2>AstroBot</h2><p>{escape(message)}</p>"
        "<p><a href='/auth/login'>Sign in with Google</a></p></body>"
    ).encode()
    headers = [(b"content-type", b"text/html; charset=utf-8"),
               (b"content-length", str(len(body)).encode()), (b"cache-control", b"no-store")]
    headers += [(b"set-cookie", c.encode()) for c in cookies or []]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _reject(scope: dict[str, Any], send: Callable, status: int, message: str):
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    body = json.dumps({"detail": message}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})
