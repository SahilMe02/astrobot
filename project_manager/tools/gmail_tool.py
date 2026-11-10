"""
Gmail tools for the AstroBot Project Manager.

Production Gmail access uses an OAuth authorized-user credential stored in
Google Secret Manager. The current project already has the OAuth credential
format required by google-auth:

    token
    refresh_token
    client_id
    client_secret
    token_uri
    scopes

The Project Manager remains client-scoped and read-only.

"""

from __future__ import annotations

import base64
import io
import json
import logging
import os

from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv

from tenancy import registry
from tenancy.context import active_client

load_dotenv()

logger = logging.getLogger("project-manager-gmail")


# ============================================================
# CONFIGURATION
# ============================================================

GCP_PROJECT = (
    os.getenv("GCP_PROJECT")
    or os.getenv("GOOGLE_CLOUD_PROJECT", "")
)

GMAIL_ACCOUNT = os.getenv(
    "ASTROBOT_GMAIL_ACCOUNT",
    "",
)

# Production OAuth user credential secret.
#
# IMPORTANT:
# Do NOT use ASTROBOT_PM_SA_SECRET_ID here.
# That variable was for the previous service-account implementation.
GMAIL_TOKEN_SECRET_ID = os.getenv(
    "ASTROBOT_GMAIL_TOKEN_SECRET_ID",
    "astrobot-gmail-token",
)

GMAIL_BASE = (
    "https://gmail.googleapis.com/gmail/v1/users/me"
)

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def _configuration_error() -> str | None:
    """Return a configuration error when required settings are missing."""

    missing: list[str] = []

    if not GCP_PROJECT:
        missing.append(
            "GCP_PROJECT/GOOGLE_CLOUD_PROJECT"
        )

    if not GMAIL_ACCOUNT:
        missing.append(
            "ASTROBOT_GMAIL_ACCOUNT"
        )

    if missing:
        return (
            "Project Manager Gmail is not configured yet. "
            "Missing: "
            + ", ".join(missing)
            + ". Configure these values before using email tools."
        )

    return None


# ============================================================
# SECRET MANAGER / OAUTH CREDENTIALS
# ============================================================

def _load_oauth_credentials_info() -> dict[str, Any]:
    """
    Load the OAuth authorized-user credential JSON from Secret Manager.

    The secret must contain a Google OAuth authorized-user credential,
    not a service-account JSON key.
    """

    from google.cloud import secretmanager

    client = (
        secretmanager.SecretManagerServiceClient()
    )

    name = (
        f"projects/{GCP_PROJECT}"
        f"/secrets/{GMAIL_TOKEN_SECRET_ID}"
        "/versions/latest"
    )

    response = client.access_secret_version(
        request={"name": name}
    )

    raw_value = (
        response.payload.data
        .decode("utf-8")
        .strip()
    )

    if not raw_value:
        raise RuntimeError(
            f"Secret '{GMAIL_TOKEN_SECRET_ID}' "
            "is empty."
        )

    try:
        data = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Secret '{GMAIL_TOKEN_SECRET_ID}' "
            "does not contain valid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Secret '{GMAIL_TOKEN_SECRET_ID}' "
            "must contain a JSON object."
        )

    return data


def _get_gmail_session():
    """
    Build an authenticated Gmail session using OAuth user credentials.

    The OAuth refresh token allows the access token to be refreshed
    automatically without interactive browser authentication.
    """

    from google.auth.transport.requests import (
        AuthorizedSession,
        Request,
    )
    from google.oauth2.credentials import (
        Credentials,
    )

    credential_info = (
        _load_oauth_credentials_info()
    )

    required_fields = {
        "client_id",
        "client_secret",
        "token_uri",
    }

    missing_fields = sorted(
        field
        for field in required_fields
        if not credential_info.get(field)
    )

    if missing_fields:
        raise RuntimeError(
            "OAuth credential is incomplete. "
            "Missing fields: "
            + ", ".join(missing_fields)
        )

    if not credential_info.get(
        "refresh_token"
    ):
        raise RuntimeError(
            "OAuth credential does not contain a "
            "refresh_token. Production Gmail access "
            "requires a refresh token."
        )

    credentials = Credentials.from_authorized_user_info(
        credential_info,
        scopes=GMAIL_SCOPES,
    )

    # Refresh the access token when required.
    if credentials.expired:
        logger.info(
            "Gmail OAuth access token expired; "
            "refreshing using refresh token."
        )

        credentials.refresh(
            Request()
        )

    if not credentials.valid:
        raise RuntimeError(
            "Gmail OAuth credentials are invalid "
            "after refresh."
        )

    return AuthorizedSession(
        credentials
    )


# ============================================================
# GMAIL HTTP HELPERS
# ============================================================

def _gmail_get(
    session,
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform a Gmail API GET request."""

    response = session.get(
        f"{GMAIL_BASE}/{path}",
        params=params or {},
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Gmail API "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


# ============================================================
# CLIENT AUTHORIZATION
# ============================================================

def _authorized_client(
    client_id: str,
    tool_context: Any = None,
) -> tuple[bool, str]:
    """
    Enforce the client boundary.

    The client is the server-set active client (tenancy.context). The
    model-written client_id argument can only match it, never change it.
    """
    active = active_client()
    requested = str(client_id or "").strip().lower()

    if not active:
        return False, "A client context is required before accessing project email."

    if requested and requested != active:
        return False, (
            f"Access denied: session is locked to client '{active}', "
            f"not '{requested}'."
        )

    return True, ""


# ============================================================
# CLIENT MAILBOX SCOPE
# ============================================================

_PM_THREADS_KEY = "_pm_allowed_threads"


def _client_mail_query(client_id: str) -> str:
    """Server-owned Gmail filter for the client (registry: gmail.query)."""
    gmail = (registry.clients().get(client_id) or {}).get("gmail") or {}
    query = str(gmail.get("query", "")).strip()
    if not query:
        raise PermissionError(
            f"Project email is not configured for client '{client_id}'."
        )
    return query


def _scoped_query(client_id: str, extra: str = "") -> str:
    """Client filter AND optional model-supplied terms.

    The extra terms are grouped and stripped of brackets, so they can only
    narrow the client's mailbox scope, never widen it.
    """
    base = f"({_client_mail_query(client_id)})"
    extra = "".join(ch for ch in str(extra or "") if ch not in "(){}").strip()
    return f"{base} ({extra})" if extra else base


def _remember_threads(tool_context: Any, thread_ids: list[str]) -> None:
    if tool_context is None:
        return
    allowed = dict(tool_context.state.get(_PM_THREADS_KEY) or {})
    known = set(allowed.get(active_client(), []))
    known.update(t for t in thread_ids if t)
    allowed[active_client()] = sorted(known)
    tool_context.state[_PM_THREADS_KEY] = allowed


def _thread_allowed(tool_context: Any, thread_id: str) -> bool:
    """Only threads returned by this client's scoped search may be opened."""
    if tool_context is None or not thread_id:
        return False
    allowed = (tool_context.state.get(_PM_THREADS_KEY) or {}).get(active_client(), [])
    return thread_id in allowed


_THREAD_DENIED = {
    "error": "Access denied: this email thread is not in the current client's mailbox scope."
}


def _require_ready(
    client_id: str,
    tool_context: Any = None,
) -> dict[str, Any] | None:
    """Validate client context and Gmail configuration."""

    allowed, reason = _authorized_client(
        client_id,
        tool_context,
    )

    if not allowed:
        return {
            "error": reason
        }

    config_error = _configuration_error()

    if config_error:
        return {
            "error": config_error
        }

    return None


# ============================================================
# FETCH RECENT THREADS
# ============================================================

def fetch_recent_threads(
    client_id: str,
    max_results: int = 20,
    query: str = "",
    user_email: str = "",
    tool_context=None,
) -> dict[str, Any]:
    """Fetch recent Gmail threads relevant to the active client."""

    error = _require_ready(
        client_id,
        tool_context,
    )

    if error:
        return error

    try:
        session = _get_gmail_session()

        client_id = active_client()
        search_query = _scoped_query(client_id, query)

        result = _gmail_get(
            session,
            "threads",
            params={
                "maxResults": int(
                    max_results
                ),
                "q": search_query,
            },
        )

        threads = result.get(
            "threads",
            [],
        )

        _remember_threads(
            tool_context,
            [t.get("id", "") for t in threads],
        )

        return {
            "client_id": client_id,
            "user_email": user_email,
            "threads": threads,
            "count": len(threads),
        }

    except Exception as exc:
        logger.exception(
            "fetch_recent_threads failed"
        )

        return {
            "error": str(exc)
        }


# ============================================================
# THREAD SNIPPETS
# ============================================================

def extract_thread_snippets(
    thread_id: str,
    client_id: str,
    user_email: str = "",
    tool_context=None,
) -> dict[str, Any]:
    """Return subject/sender/date/snippet metadata."""

    error = _require_ready(
        client_id,
        tool_context,
    )

    if error:
        return error

    if not _thread_allowed(tool_context, thread_id):
        return dict(_THREAD_DENIED)

    try:
        session = _get_gmail_session()

        thread = _gmail_get(
            session,
            f"threads/{thread_id}",
            params={
                "format": "metadata",
                "metadataHeaders": [
                    "Subject",
                    "From",
                    "Date",
                ],
            },
        )

        snippets: list[
            dict[str, str]
        ] = []

        for message in thread.get(
            "messages",
            [],
        ):
            headers = {
                h.get("name", ""):
                h.get("value", "")
                for h in message.get(
                    "payload",
                    {},
                ).get(
                    "headers",
                    [],
                )
            }

            snippets.append(
                {
                    "message_id": message.get(
                        "id",
                        "",
                    ),
                    "subject": headers.get(
                        "Subject",
                        "(no subject)",
                    ),
                    "from": headers.get(
                        "From",
                        "",
                    ),
                    "date": headers.get(
                        "Date",
                        "",
                    ),
                    "snippet": message.get(
                        "snippet",
                        "",
                    ),
                }
            )

        return {
            "client_id": client_id,
            "thread_id": thread_id,
            "snippets": snippets,
        }

    except Exception as exc:
        logger.exception(
            "extract_thread_snippets failed"
        )

        return {
            "error": str(exc)
        }


# ============================================================
# FULL THREAD MESSAGES
# ============================================================

def get_thread_messages_full(
    thread_id: str,
    client_id: str,
    user_email: str = "",
    tool_context=None,
) -> dict[str, Any]:
    """Return plain-text bodies of messages in a Gmail thread."""

    error = _require_ready(
        client_id,
        tool_context,
    )

    if error:
        return error

    if not _thread_allowed(tool_context, thread_id):
        return dict(_THREAD_DENIED)

    try:
        session = _get_gmail_session()

        thread = _gmail_get(
            session,
            f"threads/{thread_id}",
            params={
                "format": "full"
            },
        )

        messages: list[
            dict[str, str]
        ] = []

        def walk_parts(
            parts: list[
                dict[str, Any]
            ]
        ) -> str:
            for part in parts:
                mime_type = part.get(
                    "mimeType",
                    "",
                )

                if mime_type == "text/plain":
                    data = (
                        part.get(
                            "body",
                            {},
                        ).get(
                            "data",
                            "",
                        )
                    )

                    if data:
                        return (
                            base64.urlsafe_b64decode(
                                data
                            )
                            .decode(
                                "utf-8",
                                errors="ignore",
                            )
                        )

                nested = part.get(
                    "parts",
                    [],
                )

                if nested:
                    nested_body = walk_parts(
                        nested
                    )

                    if nested_body:
                        return nested_body

            return ""

        for message in thread.get(
            "messages",
            [],
        ):
            payload = message.get(
                "payload",
                {},
            )

            headers = {
                h.get("name", ""):
                h.get("value", "")
                for h in payload.get(
                    "headers",
                    [],
                )
            }

            body = walk_parts(
                payload.get(
                    "parts",
                    [],
                )
            )

            if not body:
                data = (
                    payload.get(
                        "body",
                        {},
                    ).get(
                        "data",
                        "",
                    )
                )

                if data:
                    body = (
                        base64.urlsafe_b64decode(
                            data
                        )
                        .decode(
                            "utf-8",
                            errors="ignore",
                        )
                    )

            messages.append(
                {
                    "message_id": message.get(
                        "id",
                        "",
                    ),
                    "subject": headers.get(
                        "Subject",
                        "",
                    ),
                    "from": headers.get(
                        "From",
                        "",
                    ),
                    "date": headers.get(
                        "Date",
                        "",
                    ),
                    "body": body.strip(),
                }
            )

        return {
            "client_id": client_id,
            "thread_id": thread_id,
            "messages": messages,
        }

    except Exception as exc:
        logger.exception(
            "get_thread_messages_full failed"
        )

        return {
            "error": str(exc)
        }


# ============================================================
# ATTACHMENTS
# ============================================================

def get_message_attachments(
    message_id: str,
    client_id: str,
    user_email: str = "",
    tool_context=None,
) -> dict[str, Any]:
    """
    Fetch attachment metadata and extract text
    for common document types.
    """

    error = _require_ready(
        client_id,
        tool_context,
    )

    if error:
        return error

    try:
        session = _get_gmail_session()

        message = _gmail_get(
            session,
            f"messages/{message_id}",
            params={
                "format": "full"
            },
        )

        if not _thread_allowed(tool_context, message.get("threadId", "")):
            return dict(_THREAD_DENIED)

        attachments: list[
            dict[str, Any]
        ] = []

        def extract_text(
            filename: str,
            data: bytes,
        ) -> str:
            extension = os.path.splitext(
                filename.lower()
            )[1]

            if extension in {
                ".txt",
                ".csv",
                ".json",
                ".xml",
                ".html",
                ".htm",
                ".md",
                ".log",
                ".yaml",
                ".yml",
                ".tsv",
            }:
                return data.decode(
                    "utf-8",
                    errors="ignore",
                )

            if extension == ".pdf":
                try:
                    import pypdf

                    reader = (
                        pypdf.PdfReader(
                            io.BytesIO(data)
                        )
                    )

                    return "\n".join(
                        page.extract_text() or ""
                        for page in reader.pages
                    )

                except Exception as exc:
                    return (
                        f"[PDF read error: {exc}]"
                    )

            return (
                "[Attachment content extraction "
                f"not configured for "
                f"{extension or 'this file type'}]"
            )

        def walk_parts(
            parts: list[
                dict[str, Any]
            ]
        ) -> None:
            for part in parts:

                if part.get("parts"):
                    walk_parts(
                        part["parts"]
                    )
                    continue

                filename = part.get(
                    "filename",
                    "",
                )

                attachment_id = (
                    part.get(
                        "body",
                        {},
                    ).get(
                        "attachmentId"
                    )
                )

                if (
                    not filename
                    or not attachment_id
                ):
                    continue

                data = _gmail_get(
                    session,
                    (
                        f"messages/{message_id}"
                        f"/attachments/{attachment_id}"
                    ),
                ).get(
                    "data",
                    "",
                )

                raw = (
                    base64.urlsafe_b64decode(
                        data
                    )
                )

                attachments.append(
                    {
                        "filename": filename,
                        "mime_type": part.get(
                            "mimeType",
                            "",
                        ),
                        "size_bytes": part.get(
                            "body",
                            {},
                        ).get(
                            "size",
                            0,
                        ),
                        "content": extract_text(
                            filename,
                            raw,
                        ),
                    }
                )

        walk_parts(
            message.get(
                "payload",
                {},
            ).get(
                "parts",
                [],
            )
        )

        return {
            "client_id": client_id,
            "message_id": message_id,
            "attachments": attachments,
        }

    except Exception as exc:
        logger.exception(
            "get_message_attachments failed"
        )

        return {
            "error": str(exc)
        }


# ============================================================
# DATE-SCOPED EMAIL SEARCH
# ============================================================

def fetch_emails_by_date(
    date_expression: str,
    client_id: str,
    user_email: str = "",
    tool_context=None,
) -> dict[str, Any]:
    """Fetch emails for a simple date expression."""

    error = _require_ready(
        client_id,
        tool_context,
    )

    if error:
        return error

    expr = str(
        date_expression or ""
    ).strip()

    today = datetime.now(
        timezone.utc
    ).date()

    resolved = None

    for fmt in (
        "%Y-%m-%d",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
    ):
        try:
            resolved = datetime.strptime(
                expr,
                fmt,
            ).date()

            break

        except ValueError:
            pass

    if (
        resolved is None
        and expr.lower() == "yesterday"
    ):
        resolved = (
            today
            - timedelta(days=1)
        )

    if resolved is None:
        resolved = today

    date_str = (
        resolved
        .isoformat()
        .replace("-", "/")
    )

    next_str = (
        (
            resolved
            + timedelta(days=1)
        )
        .isoformat()
        .replace("-", "/")
    )

    return fetch_recent_threads(
        client_id=client_id,
        max_results=20,
        query=(
            f"after:{date_str} "
            f"before:{next_str}"
        ),
        user_email=user_email,
        tool_context=tool_context,
    )
def collect_recent_project_email_context(
    client_id: str,
    user_email: str = "",
    max_results: int = 20,
    query: str = "",
    tool_context=None,
    **kwargs,
) -> dict[str, Any]:
    """
    Backward-compatible Project Manager Gmail helper.

    Collects recent client email threads and lightweight thread metadata
    while preserving the existing client-scoped Gmail workflow.
    """

    error = _require_ready(
        client_id,
        tool_context,
    )

    if error:
        return error

    recent = fetch_recent_threads(
        client_id=client_id,
        max_results=max_results,
        query=query,
        user_email=user_email,
        tool_context=tool_context,
    )

    if recent.get("error"):
        return recent

    threads = recent.get(
        "threads",
        [],
    )

    context_threads: list[dict[str, Any]] = []

    for thread in threads:
        thread_id = thread.get(
            "id",
            "",
        )

        if not thread_id:
            continue

        snippets = extract_thread_snippets(
            thread_id=thread_id,
            client_id=client_id,
            user_email=user_email,
            tool_context=tool_context,
        )

        context_threads.append(
            {
                "thread_id": thread_id,
                "snippets": snippets.get(
                    "snippets",
                    [],
                ),
            }
        )

    return {
        "client_id": client_id,
        "user_email": user_email,
        "count": len(threads),
        "threads": threads,
        "thread_context": context_threads,
    }