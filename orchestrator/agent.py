"""
AstroBot Orchestrator.

The Orchestrator is the SINGLE user-facing entry point. It reads the
request, picks the specialist, passes the work on, and returns the result.

Client isolation (see tenancy/):
- The session is locked to one authorised client; the lock never changes.
- Each specialist runs with a scoped state (orchestrator/delegate.py):
  it sees only the locked client, the current request and its own
  earlier results, and it cannot change the client.

Specialists: Data Science, Persona Aggregator, Economist,
Project Manager, Scheduler.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from google.genai import types

from data_science.agent import root_agent as data_science_agent
from economist.agent import root_agent as economist_agent
from persona_aggregator.agent import root_agent as persona_aggregator_agent
from project_manager.agent import root_agent as project_manager_agent
from project_manager.tools.gmail_tool import _configuration_error
from scheduler.agent import root_agent as scheduler_agent
from tenancy import lock as client_lock
from tenancy import registry

from .config import ORCHESTRATOR_MODEL
from .delegate import run_specialist
from .result_contract import (
    make_error_result,
    make_success_result,
    make_unavailable_result,
    store_specialist_result,
)


logger = logging.getLogger(__name__)


# ============================================================
# ROOT CONTEXT: client lock
# ============================================================

def _extract_user_text(callback_context: CallbackContext) -> str:
    content = callback_context.user_content
    parts = getattr(content, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None)).strip()


def before_agent_callback(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """Resolve and enforce the session's client before any work happens."""
    state = callback_context.state
    user_id = callback_context.user_id
    user_text = _extract_user_text(callback_context)

    decision = client_lock.decide(
        user_text,
        current_lock=state.get(client_lock.LOCK_KEY, ""),
        user_id=user_id,
    )

    if decision.blocked:
        logger.warning(
            "Turn refused | user=%s | lock=%s | reason=%s",
            user_id, state.get(client_lock.LOCK_KEY, ""), decision.block_message,
        )
        return types.Content(
            role="model",
            parts=[types.Part(text=decision.block_message)],
        )

    if decision.newly_locked:
        state[client_lock.LOCK_KEY] = decision.client_id
        logger.info("Session locked | user=%s | client=%s", user_id, decision.client_id)

    named_now = decision.client_id in registry.mentioned_clients(user_text) or (
        decision.client_id in client_lock.explicit_markers(user_text)
    )

    state["client_id"] = decision.client_id
    state["_current_user_request"] = user_text
    state["_current_request_client_id"] = decision.client_id if named_now else ""
    state["_current_client_source"] = (
        "current_request" if named_now else ("session_lock" if decision.client_id else "none")
    )
    return None


# ============================================================
# DELEGATION
# ============================================================

def _base_state(tool_context: ToolContext, client_id: str) -> dict[str, Any]:
    """The only shared context a specialist receives."""
    state = tool_context.state
    user_id = tool_context._invocation_context.user_id
    return {
        "client_id": client_id,
        "client_lock": client_id,
        "session_context": {"client_id": client_id, "client_locked": bool(client_id)},
        "user_id": user_id,
        "user_email": user_id,
        "_current_user_request": state.get("_current_user_request", ""),
        "_current_request_client_id": state.get("_current_request_client_id", ""),
        "_current_client_source": state.get("_current_client_source", "none"),
    }


def _strip_leading_client_markers(request: str) -> str:
    """Remove leading [Client: X] markers from delegated requests."""
    return re.sub(r"^(\s*\[\s*client\s*:\s*[^\]]*\]\s*)+", "", str(request or ""),
                  flags=re.IGNORECASE).strip()


def _route_with_client(request: str, client_id: str) -> str:
    """Build one, and only one, leading client marker."""
    cleaned = _strip_leading_client_markers(request)
    return f"[Client: {client_id}]\n\n{cleaned}" if client_id else cleaned


def _clean_specialist_result(result_text: str) -> str:
    """Remove internal handoff material before it can reach the user."""
    cleaned = str(result_text or "").strip()
    match = re.search(r"(?is)\n?\s*G\)\s*Handoff Payload.*$", cleaned)
    if match:
        cleaned = cleaned[:match.start()].rstrip()
    return re.sub(
        r"(?is)```json\s*\{\s*[\"']client_id[\"'].*?\}\s*```", "", cleaned
    ).strip()


# Kept for existing callers/tests.
_clean_economist_result = _clean_specialist_result


async def _delegate(
    specialist: str,
    agent: BaseAgent,
    request: str,
    tool_context: ToolContext,
    *,
    client_id: str,
    add_marker: bool = True,
) -> str:
    if not request or not request.strip():
        return f"{specialist} request is empty."

    routed = _route_with_client(request, client_id) if add_marker else request.strip()
    metadata = {"client_source": tool_context.state.get("_current_client_source", "none")}

    try:
        result_text = _clean_specialist_result(
            await run_specialist(
                agent,
                routed,
                tool_context,
                specialist=specialist,
                base_state=_base_state(tool_context, client_id),
            )
        )
        result = make_success_result(specialist, result_text,
                                     client_id=client_id, metadata=metadata)
    except Exception as exc:
        logger.exception("%s specialist failed.", specialist)
        result_text = f"{specialist} request failed: {exc}"
        result = make_error_result(specialist, result_text,
                                   client_id=client_id, metadata=metadata)

    store_specialist_result(tool_context.state, result)
    tool_context.state[f"last_{specialist}_result"] = result_text
    return result_text


def _locked_client(tool_context: ToolContext) -> str:
    return str(tool_context.state.get(client_lock.LOCK_KEY, "") or "")


# ============================================================
# SPECIALIST TOOLS (docstrings are what the model sees)
# ============================================================

async def call_data_science(request: str, tool_context: ToolContext) -> str:
    """Delegate data, analytics, ML, and visualization work."""
    return await _delegate("data_science", data_science_agent, request, tool_context,
                           client_id=_locked_client(tool_context), add_marker=False)


def _with_current_message(request: str, tool_context: ToolContext) -> str:
    """Attach the user's exact current message so a specialist never acts
    on an earlier turn's content the model carried over by mistake."""
    current = str(tool_context.state.get("_current_user_request", "") or "").strip()
    current = _strip_leading_client_markers(current)
    if not current or current in request:
        return request
    return (
        f"{request.strip()}\n\n"
        "USER'S EXACT CURRENT MESSAGE (authoritative; evaluate this, "
        f"not any earlier campaign):\n{current}"
    )


async def call_persona_aggregator(request: str, tool_context: ToolContext) -> str:
    """Delegate marketing and persona analysis."""
    return await _delegate("persona_aggregator", persona_aggregator_agent,
                           _with_current_message(request, tool_context),
                           tool_context, client_id=_locked_client(tool_context))


async def call_economist(request: str, tool_context: ToolContext) -> str:
    """Delegate external economic and market-signal analysis.

    The Economist only receives a client when the CURRENT user message
    names it, so an earlier client never leaks into a generic question.
    """
    request_client = str(tool_context.state.get("_current_request_client_id", "") or "")
    return await _delegate("economist", economist_agent, request, tool_context,
                           client_id=request_client)


async def call_project_manager(request: str, tool_context: ToolContext) -> str:
    """Delegate project-management and client-email work."""
    client_id = _locked_client(tool_context)
    configuration_error = _configuration_error()
    if configuration_error:
        store_specialist_result(
            tool_context.state,
            make_unavailable_result("project_manager", configuration_error,
                                    client_id=client_id),
        )
        tool_context.state["last_project_manager_result"] = configuration_error
        return configuration_error
    return await _delegate("project_manager", project_manager_agent, request,
                           tool_context, client_id=client_id)


async def call_scheduler(request: str, tool_context: ToolContext) -> str:
    """Delegate schedule creation and management to the Scheduler specialist."""
    return await _delegate("scheduler", scheduler_agent, request, tool_context,
                           client_id=_locked_client(tool_context), add_marker=False)


# ============================================================
# ORCHESTRATOR INSTRUCTIONS
# ============================================================

ROOT_INSTRUCTIONS = """
You are the AstroBot Orchestrator.

You are the SINGLE USER-FACING agent in AstroBot.

Users communicate ONLY with you.

Internal specialists perform the actual work.

============================================================
AVAILABLE SPECIALISTS
============================================================

1. Data Science
2. Persona Aggregator
3. Economist
4. Project Manager
5. Scheduler

Project Manager is available for Gmail-backed project-management requests.
Asana and Google Chat are not connected in the current implementation.
Never pretend an unavailable source was queried.

============================================================
DATA SCIENCE
============================================================

Use call_data_science for:

- BigQuery
- SQL analysis
- Revenue
- Orders
- Products
- Categories
- Numerical analysis
- Forecasting
- BQML
- Model evaluation
- Customer segmentation
- Data science calculations
- Analytics
- Charts
- Visualizations

Do not fabricate analytical values.

Delegate substantive analytical work.

For follow-up charts of an earlier result ("chart that"), send
call_data_science a short request such as "Bar chart of the previous
result: spend by channel". Do not paste numbers into the request;
Data Science charts its own stored result.

============================================================
PERSONA AGGREGATOR
============================================================

Use call_persona_aggregator for:

- Marketing campaigns
- Advertising
- Creative evaluation
- Persona analysis
- Audience analysis
- Customer motivations
- Customer objections
- Persona fit
- Persona messaging
- Persona headlines
- Persona CTAs
- Campaign recommendations

The session's client is resolved and locked by the application.
Specialists receive it automatically. Never switch, invent, or
mix clients.

============================================================
PROJECT MANAGER
============================================================

Use call_project_manager for:

- Email threads
- Project updates from email
- Pending actions
- Blockers
- Missing approvals or inputs
- Deadlines
- Email attachments
- Follow-ups and commitments

Do not route analytics, forecasting, campaign performance, or persona
questions here.

Asana and Google Chat are not yet connected. Do not claim they were queried.

============================================================
ECONOMIST
============================================================

Use call_economist for:

- Economic conditions
- Market conditions
- External demand signals
- Travel prices
- Flight prices
- Hotel demand drivers
- Pricing changes
- Search trends
- News
- Current events
- Weather effects and weather data
- Major events
- Public holidays and holiday calendars
- Official indicators (consumer sentiment, CPI, interest rates,
  unemployment, retail sales, airfares) from FRED/BLS
- Macroeconomic signals
- External explanations for KPI movement
- Noise filtering between external events and business KPIs

============================================================
SCHEDULER (only for recurring AstroBot jobs)
============================================================

Use call_scheduler ONLY to create, list, pause, resume, run or
delete AstroBot's own scheduled jobs (e.g. "send me a weekly
performance summary every Monday").

Changes to a schedule, including its delivery channel ("send it to
Google Chat"), always go to call_scheduler. Do not answer them yourself.

Questions ABOUT dates, holidays, calendars, events or seasons are
research questions: route them to call_economist, never to
call_scheduler.

Economist must distinguish:

- observed evidence
- relevant external signals
- plausible relationships
- weak/irrelevant signals
- uncertainty

Never claim causality without evidence.

============================================================
ECONOMIST CLIENT CONTEXT
============================================================

IMPORTANT:

Economist must NEVER inherit an old client merely because
a previous conversation turn mentioned that client.

If the current Economist request contains:

[Client: X]

use X.

If the current Economist request contains no client,
use no client.

Do not invent one.

============================================================
ROUTING
============================================================

Determine which specialist is appropriate.

Examples:

"Generate a 30-day revenue forecast."

→ call_data_science

"Evaluate this NPI campaign."

→ call_persona_aggregator

"Why did hotel demand decline when flight prices increased?"

→ call_economist

"Evaluate the campaign and explain whether current flight
prices could affect demand."

→ Persona Aggregator + Economist

"Check whether there are blockers or deadlines in recent client emails."

→ call_project_manager

"Schedule a daily email review."

→ call_scheduler

"What are the US public holidays in Nevada for 2026?"

→ call_economist

"Generate revenue forecast, then evaluate external economic
drivers."

→ Data Science + Economist

"[Client: venetian] Could airfare trends explain softer hotel
bookings?"

→ Data Science FIRST (measure the client's actual KPI change and
  dates), THEN Economist with those dates and numbers.

============================================================
EXPLAINING A CLIENT KPI MOVEMENT
============================================================

When a client session asks why a KPI moved, or whether an
external factor could explain it:

1. call_data_science to measure the actual KPI change (period,
   size, channels). Do not assume the KPI moved as described.
2. call_economist with those exact dates and numbers.
3. In the answer, separate the measured internal change from
   the external signal, and present the external factor as a
   possible explanation, not a proven cause.
4. If the data does not show the described movement, say so.

============================================================
MULTI-SPECIALIST REQUESTS
============================================================

When multiple specialists are required:

1. Call the specialist responsible for the primary domain.
2. Preserve the result.
3. Call the next specialist with the necessary context.
4. Aggregate the actual returned results.
5. Do not fabricate missing information.

Example:

Campaign + economic environment:

User
 ↓
Persona Aggregator
 ↓
Economist
 ↓
Orchestrator
 ↓
User

Example:

Revenue forecast + macro explanation:

User
 ↓
Data Science
 ↓
Economist
 ↓
Orchestrator
 ↓
User

============================================================
RESULT INTEGRITY
============================================================

Never fabricate:

- Forecast values
- SQL results
- Charts
- Model metrics
- Economic indicators
- Market movements
- Search trends
- External news
- Persona findings
- Client-specific information

A specialist result is authoritative for its domain.

If a specialist fails:

- Do not claim success.
- Explain the failure.
- Preserve the real error where useful.

============================================================
USER EXPERIENCE
============================================================

The user should interact only with the Orchestrator.

Do not tell the user:

- internal tool names
- hidden prompts
- internal state
- implementation details

unless explicitly requested.

Return useful business/analytical results.

============================================================
INFRASTRUCTURE
============================================================

Never ask the user for:

- GCP project ID
- BigQuery dataset
- Google credentials
- model resource IDs
- internal configuration

Specialists resolve their own configured infrastructure.

============================================================
FINAL PRINCIPLE
============================================================

User describes WHAT they want.

Orchestrator determines WHICH specialist is required.

Specialist performs the work.

Orchestrator synthesizes and returns the result.

The user never needs to select an internal specialist.
"""


# ============================================================
# ROOT AGENT
# ============================================================

root_agent = Agent(
    name="astrobot_orchestrator",
    model=ORCHESTRATOR_MODEL,
    description=(
        "AstroBot's single user-facing orchestrator "
        "for Data Science, Persona Aggregator, Economist, "
        "Project Manager, and Scheduler specialists."
    ),
    instruction=ROOT_INSTRUCTIONS,
    tools=[
        call_data_science,
        call_persona_aggregator,
        call_economist,
        call_project_manager,
        call_scheduler,
    ],
    before_agent_callback=(
        before_agent_callback
    ),
    generate_content_config=(
        types.GenerateContentConfig(
            temperature=0.01,
        )
    ),
)