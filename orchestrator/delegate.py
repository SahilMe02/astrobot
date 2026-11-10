"""Run one specialist with a scoped view of session state.

Replaces ADK's AgentTool.run_async for Orchestrator -> specialist calls.
AgentTool copies the whole parent state into the child and copies every
child state change back. That lets any specialist read other specialists'
results and overwrite the session's client.

Here each specialist sees only:
  * the trusted base context the Orchestrator passes in, and
  * its own private state from earlier turns.
Changes it makes are saved to its private slot only. Protected keys
(client, user, current request) can never be written by a specialist.

The specialist also runs inside tenancy.context.client_scope, which is
what data-access code uses to pick the client's tables.
"""

from __future__ import annotations

import logging
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools import ToolContext
from google.adk.tools._forwarding_artifact_service import ForwardingArtifactService
from google.adk.utils.context_utils import Aclosing
from google.genai import types

from tenancy.context import client_scope

logger = logging.getLogger(__name__)

PROTECTED_KEYS = frozenset({
    "client_id",
    "client_lock",
    "client_locked",
    "LOCKED_CLIENT",
    "session_context",
    "user_id",
    "user_email",
    "_current_user_request",
    "_current_request_client_id",
    "_current_client_source",
})

_SHARED_PREFIXES = ("app:", "user:", "temp:", "_adk")


def private_state_key(specialist: str) -> str:
    return f"_specialist__{specialist}"


def _keep(key: str) -> bool:
    return key not in PROTECTED_KEYS and not key.startswith(_SHARED_PREFIXES)


async def run_specialist(
    agent: BaseAgent,
    request: str,
    tool_context: ToolContext,
    *,
    specialist: str,
    base_state: dict[str, Any],
) -> str:
    """Run `agent` on `request` and return its final text."""
    slot = private_state_key(specialist)
    private: dict[str, Any] = dict(tool_context.state.get(slot) or {})

    invocation = tool_context._invocation_context
    app_name = invocation.app_name or agent.name

    runner = Runner(
        app_name=app_name,
        agent=agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=invocation.credential_service,
    )

    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=invocation.user_id,
        state={**private, **base_state},
    )

    message = types.Content(role="user", parts=[types.Part.from_text(text=request)])
    last_content = None

    try:
        with client_scope(base_state.get("client_lock", "")):
            async with Aclosing(
                runner.run_async(
                    user_id=session.user_id,
                    session_id=session.id,
                    new_message=message,
                )
            ) as events:
                async for event in events:
                    for key, value in (event.actions.state_delta or {}).items():
                        if _keep(key):
                            private[key] = value
                        elif key in PROTECTED_KEYS and value != base_state.get(key):
                            logger.warning(
                                "Ignored write to protected key %r by %s",
                                key, specialist,
                            )
                    if event.content:
                        last_content = event.content
    finally:
        await runner.close()
        tool_context.state[slot] = private

    if not last_content or not last_content.parts:
        return ""
    return "\n".join(
        p.text for p in last_content.parts if p.text and not p.thought
    )
