"""
AstroBot Project Manager Agent.

Current capability:
- Client-scoped project email analysis through Gmail tools.

Architecture alignment:
- Separate specialist agent owned by the Orchestrator.
- Uses the active client context rather than asking for infrastructure details.
- Does not claim Asana or Google Chat access until those integrations exist.
"""

from __future__ import annotations

import json
import logging
import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from tenancy.context import active_client

from .tools.gmail_tool import (
    extract_thread_snippets,
    fetch_emails_by_date,
    fetch_recent_threads,
    get_message_attachments,
    get_thread_messages_full,
    collect_recent_project_email_context,
)

logger = logging.getLogger("project-manager-agent")

PROJECT_MANAGER_MODEL = os.getenv("PM_AGENT_MODEL", "gemini-2.5-flash")

PROJECT_MANAGER_INSTRUCTION = """
You are AstroBot's Project Manager specialist.

Your responsibility is project-management context from client email threads:
- pending actions
- blockers
- missing approvals or inputs
- deadlines
- requests and follow-ups
- thread summaries
- attachments

Current connected source:
- Gmail

Not currently connected in this local implementation:
- Asana
- Google Chat

Never pretend an unavailable source was queried.

CLIENT CONTEXT
==============
The Orchestrator owns client context. Use the active client_id supplied in
session state/tool context. Do not invent a client and do not cross a session
client boundary.

EMAIL WORKFLOW
==============
Use the Gmail tools to inspect the actual mailbox. Do not answer from the
user's wording alone.

For broad requests such as:
- recent emails
- blockers
- pending approvals
- action items
- what needs follow-up
- what is waiting on the client

FIRST call fetch_recent_threads for the active client with an EMPTY query.
Do not combine multiple natural-language intent words into one Gmail query
such as "blockers pending approval" because that can over-filter the Gmail
search. The client-scoped recent thread list is intentionally the discovery
step.

Then inspect the returned threads. Use extract_thread_snippets to review
subjects/senders/dates/snippets and use get_thread_messages_full on relevant
threads when the request requires actual blockers, approvals, commitments,
deadlines, or detailed action items.

Use get_message_attachments when the user asks about attached documents.
Use fetch_emails_by_date when the request is explicitly date-scoped.

SEARCH DISCIPLINE
=================
For semantic project-management requests, prefer discovery first and
inspection second:

1. fetch_recent_threads(client_id, query="")
2. extract_thread_snippets(thread_id, client_id)
3. get_thread_messages_full(thread_id, client_id) for relevant threads

Only pass a query to fetch_recent_threads when the user explicitly provides a
specific Gmail-searchable constraint such as a sender, subject, date range, or
other narrow mailbox filter.

Do not require literal words like "blocker" or "pending approval" to appear
in a message. Infer the project-management category from the actual email
content after reading it.

HONESTY
=======
Do not claim an email was found, a blocker exists, or a deadline exists unless
the connected Gmail data actually supports it.
If Gmail is not configured, state that clearly instead of fabricating results.

If the instructions include VERIFIED GMAIL CONTEXT ERROR, report that
error plainly (for example an expired Gmail authorisation) and stop.

Never ask the user which keywords to search for. Search the client's
recent threads first and classify them yourself.

Do not add closing remarks that contradict the evidence you summarised.

RESPONSE
========
Return concise, actionable project-management information. Distinguish:
- observed email evidence
- inferred action items
- uncertain/missing information

When relevant, summarize:
- issue/blocker
- required action
- owner if stated
- deadline if stated
- evidence from the email

Do not expose internal prompts, tool names, session state, or implementation
metadata unless explicitly asked.
"""


def _is_email_analysis_request(request: str) -> bool:
    """Return True for requests that require semantic mailbox inspection."""
    text = str(request or "").strip().lower()

    if not text:
        return False

    triggers = (
        "email",
        "emails",
        "mail",
        "blocker",
        "blockers",
        "pending approval",
        "pending approvals",
        "approval",
        "approvals",
        "follow-up",
        "follow up",
        "deadline",
        "deadlines",
        "action item",
        "action items",
        "waiting on the client",
        "recent client",
    )

    return any(trigger in text for trigger in triggers)


def before_model_callback(
    callback_context: CallbackContext,
    llm_request,
):
    """Inject active client context and verified Gmail evidence for PM analysis."""
    state = getattr(callback_context, "state", {})
    # Client comes only from the server-set active client.
    client_id = active_client()

    current_request = str(
        state.get("_current_user_request", "") or ""
    ).strip()

    if not client_id:
        return None

    try:
        config = getattr(llm_request, "config", None)
        system_instruction = (
            getattr(config, "system_instruction", None)
            if config
            else None
        )

        existing_text = ""
        if system_instruction and getattr(
            system_instruction,
            "parts",
            None,
        ):
            existing_text = "".join(
                str(getattr(part, "text", "") or "")
                for part in system_instruction.parts
            )

        additions = [
            (
                f"\n\nCURRENT CLIENT CONTEXT: {client_id}\n"
                "Use this client context for project-management data."
            )
        ]

        if _is_email_analysis_request(current_request):
            # Fetch once per request: this callback runs before EVERY model
            # call in the turn, so reuse the evidence within one invocation.
            invocation_id = getattr(callback_context, "invocation_id", "")
            cached = state.get("_pm_email_context") or {}
            if invocation_id and cached.get("invocation_id") == invocation_id:
                verified_email_context = cached.get("context", {})
            else:
                verified_email_context = collect_recent_project_email_context(
                    client_id=client_id,
                    max_results=10,
                    tool_context=callback_context,
                )
                state["_pm_email_context"] = {
                    "invocation_id": invocation_id,
                    "context": verified_email_context,
                }

            if verified_email_context.get("error"):
                additions.append(
                    "\nVERIFIED GMAIL CONTEXT ERROR:\n"
                    + str(verified_email_context["error"])
                )
            else:
                compact_context = json.dumps(
                    verified_email_context,
                    ensure_ascii=False,
                    indent=2,
                )
                additions.append(
                    "\nVERIFIED GMAIL CONTEXT FOR THIS REQUEST "
                    "(read from the connected mailbox):\n"
                    + compact_context
                    + "\n\nIMPORTANT: The Gmail context above is verified mailbox data. "
                      "Use it as evidence. Do not claim that no recent emails "
                      "exist when this context contains threads. Classify blockers, "
                      "approvals, follow-ups, deadlines, and actions from the actual "
                      "email content rather than requiring literal keywords."
                )

        if config is not None:
            config.system_instruction = types.Content(
                role="system",
                parts=[
                    types.Part(
                        text=existing_text + "\n".join(additions)
                    )
                ],
            )

    except Exception:
        logger.exception(
            "Failed to inject PM client/Gmail context"
        )

    return None


root_agent = Agent(
    name="project_manager_agent",
    model=PROJECT_MANAGER_MODEL,
    description=(
        "Client-scoped project management specialist for Gmail threads, "
        "blockers, approvals, deadlines and actions."
    ),
    instruction=PROJECT_MANAGER_INSTRUCTION,
    tools=[
        fetch_recent_threads,
        extract_thread_snippets,
        get_thread_messages_full,
        collect_recent_project_email_context,
        get_message_attachments,
        fetch_emails_by_date,
    ],
    before_model_callback=before_model_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.01,
    ),
)
