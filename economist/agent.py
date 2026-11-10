"""
AstroBot Economist Bot.

Internal specialist for external economic, market, pricing,
travel, news, weather, and event signals.

The Orchestrator is the only user-facing agent.
"""

from __future__ import annotations

import logging
from datetime import date
import re
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types

from .config import (
    ECONOMIST_MODEL,
    SUPPORTED_CLIENTS,
)
from .data_tools import ECONOMIST_DATA_TOOLS
from .prompts import ECONOMIST_INSTRUCTIONS


logger = logging.getLogger(__name__)


NORMALIZED_CLIENTS = {
    str(client).strip().lower()
    for client in SUPPORTED_CLIENTS
}


def _extract_user_text(
    callback_context: CallbackContext,
) -> str:
    """Return the current request text."""

    try:
        user_content = callback_context.user_content

        if not user_content:
            return ""

        parts = (
            getattr(
                user_content,
                "parts",
                None,
            )
            or []
        )

        texts: list[str] = []

        for part in parts:
            text = getattr(
                part,
                "text",
                None,
            )

            if text:
                texts.append(
                    str(text)
                )

        return "\n".join(texts).strip()

    except Exception:
        logger.exception(
            "Economist failed to extract current user text."
        )
        return ""


def _extract_current_client(
    text: str,
) -> str:
    """
    Extract client ONLY from the current request.

    Never use previous session/client state.
    """

    if not text:
        return "n/a"

    explicit_match = re.search(
        r"\[\s*client\s*:\s*([^\]]+)\]",
        text,
        flags=re.IGNORECASE,
    )

    if explicit_match:
        client = (
            explicit_match.group(1)
            .strip()
            .lower()
        )

        return (
            client
            if client
            else "n/a"
        )

    lowered = text.lower()

    for client in sorted(
        NORMALIZED_CLIENTS,
        key=len,
        reverse=True,
    ):
        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(client)}"
            rf"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            lowered,
        ):
            return client

    return "n/a"


def before_agent_callback(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """
    Reset Economist context on EVERY turn.

    This intentionally prevents a client from an older
    request from leaking into the current request.
    """

    user_text = _extract_user_text(
        callback_context
    )

    current_client = _extract_current_client(
        user_text
    )

    # These values are deliberately overwritten on every invocation.
    # They are turn-scoped metadata, not persistent client memory.
    callback_context.state[
        "economist_context"
    ] = {
        "client_id": current_client,
        "request": user_text,
    }

    callback_context.state[
        "economist_client_id"
    ] = current_client

    callback_context.state[
        "economist_current_request"
    ] = user_text

    logger.info(
        "Economist current-turn context | client=%s",
        current_client,
    )

    return None


root_agent = Agent(
    name="economist_agent",
    model=ECONOMIST_MODEL,
    description=(
        "Economist specialist for external economic, "
        "market, pricing, travel, news, weather, "
        "event, and demand signals."
    ),
    # Today's date is added so data-tool date ranges are correct.
    instruction=lambda _ctx: (
        f"Today's date is {date.today().isoformat()}.\n\n"
        + ECONOMIST_INSTRUCTIONS
    ),
    tools=[
        # bypass_multi_tools_limit lets built-in search sit beside the
        # structured data function tools in one agent.
        GoogleSearchTool(bypass_multi_tools_limit=True),
        *ECONOMIST_DATA_TOOLS,
    ],
    before_agent_callback=(
        before_agent_callback
    ),
    generate_content_config=(
        types.GenerateContentConfig(
            temperature=0.1,
        )
    ),
)