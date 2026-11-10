"""Deliver scheduled results to a client's approved channels.

Channels are names, never URLs. Each client's Google Chat incoming-webhook
URL is a credential kept in Secret Manager; the registry names the secret
(clients.<id>.google_chat.webhook_secret). A schedule can therefore only
deliver to its own client's configured space.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from tenancy import registry

logger = logging.getLogger("astrobot-scheduler-delivery")

INTERNAL = "internal"
GOOGLE_CHAT = "google_chat"
SUPPORTED_CHANNELS = (INTERNAL, GOOGLE_CHAT)

CHAT_TEXT_LIMIT = 4000
TIMEOUT_SECONDS = 15


def _chat_secret_name(client_id: str) -> str:
    chat = (registry.clients().get(client_id) or {}).get("google_chat") or {}
    return str(chat.get("webhook_secret", "")).strip()


def channel_problem(client_id: str, channel: str) -> str | None:
    """Why this client cannot use this channel, or None if it can."""
    if channel not in SUPPORTED_CHANNELS:
        return (
            f"Channel '{channel}' is not supported. "
            f"Supported channels: {', '.join(SUPPORTED_CHANNELS)}."
        )
    if channel == GOOGLE_CHAT and not _chat_secret_name(client_id):
        return f"Google Chat delivery is not configured for client '{client_id}'."
    return None


def _webhook_url(secret_name: str) -> str:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured.")
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        request={"name": f"projects/{project}/secrets/{secret_name}/versions/latest"}
    )
    url = response.payload.data.decode("utf-8").strip()
    if not url.startswith("https://chat.googleapis.com/"):
        raise RuntimeError("Configured secret is not a Google Chat webhook URL.")
    return url


def _send_google_chat(client_id: str, title: str, text: str) -> dict[str, Any]:
    secret_name = _chat_secret_name(client_id)
    if not secret_name:
        return {"channel": GOOGLE_CHAT, "status": "not_configured"}
    body = f"*{title}*\n\n{text}"
    if len(body) > CHAT_TEXT_LIMIT:
        body = body[: CHAT_TEXT_LIMIT - 40].rstrip() + "\n\n_(truncated; full result in AstroBot)_"
    try:
        response = requests.post(
            _webhook_url(secret_name),
            json={"text": body},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return {"channel": GOOGLE_CHAT, "status": "delivered"}
    except Exception as exc:
        # Never log the webhook URL: it is a credential.
        logger.warning("Google Chat delivery failed | client=%s | %s",
                       client_id, type(exc).__name__)
        return {"channel": GOOGLE_CHAT, "status": "failed", "error": type(exc).__name__}


def deliver(client_id: str, channels: list[str], title: str, text: str) -> list[dict[str, Any]]:
    """Send `text` to each channel; returns one honest status per channel."""
    results = []
    for channel in channels or [INTERNAL]:
        if channel == INTERNAL:
            results.append({"channel": INTERNAL, "status": "stored"})
        elif channel == GOOGLE_CHAT:
            results.append(_send_google_chat(client_id, title, text))
        else:
            results.append({"channel": channel, "status": "unsupported"})
    return results
