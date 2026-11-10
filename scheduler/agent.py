from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from google.adk.agents import Agent
from google.adk.tools import ToolContext

from tenancy.context import active_client

from .delivery import INTERNAL, channel_problem
from .models import ScheduleConfig
from .store import ScheduleStore


logger = logging.getLogger(
    "astrobot-scheduler"
)

STORE = ScheduleStore()


def _context_value(
    tool_context: ToolContext | None,
    key: str,
    default: str = "",
) -> str:
    """
    Read a value from the current ADK state.

    Supports the current flattened state as well as the nested
    compatibility state used by earlier AstroBot code.
    """
    if not tool_context:
        return default

    value = tool_context.state.get(
        key
    )

    if value:
        return str(value)

    nested = (
        tool_context.state.get(
            "state"
        )
        or {}
    )

    return str(
        nested.get(
            key,
            default,
        )
        or default
    )


def _current_client_id(
    tool_context: ToolContext | None,
) -> str:
    """The server-set active client (never model text or state)."""
    return active_client()


def _current_user_email(
    tool_context: ToolContext | None,
) -> str:
    return _context_value(
        tool_context,
        "user_email",
        "unknown",
    ).strip().lower()


def _ownership_error(
    schedule: ScheduleConfig,
    client_id: str,
    user_email: str,
) -> str | None:
    """Only the schedule's owner, in the schedule's client, may change it."""
    schedule_client = str(schedule.context.get("client_id", "")).strip().lower()
    owner_email = str(schedule.context.get("user_email", "")).strip().lower()
    if (
        not client_id
        or schedule_client != client_id
        or not _known_user(user_email)
        or owner_email != user_email
    ):
        return f"Access denied for schedule {schedule.id}."
    return None


def _known_user(user_email: str) -> bool:
    return bool(user_email) and user_email != "unknown"


def _schedule_output(
    schedule: ScheduleConfig,
    operation: str,
) -> str:
    return json.dumps(
        {
            "status": "success",
            "operation": operation,
            "schedule": schedule.model_dump(
                mode="json"
            ),
        },
        indent=2,
    )


def create_schedule(
    name: str,
    prompt: str,
    cron: str | None = None,
    interval_minutes: int | None = None,
    channels: list[str] | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """
    Create a client-scoped recurring schedule.
    """
    client_id = _current_client_id(
        tool_context
    )

    user_email = _current_user_email(
        tool_context
    )

    if not client_id:
        return (
            "Error: no client identified in the current session. "
            "Specify the client first."
        )

    if not _known_user(user_email):
        return "Error: schedules need a signed-in user."

    requested_channels = [
        str(c).strip().lower() for c in (channels or [INTERNAL])
    ]
    for channel in requested_channels:
        problem = channel_problem(client_id, channel)
        if problem:
            return f"Error: {problem}"

    try:
        schedule = ScheduleConfig(
            name=name.strip(),
            prompt=prompt.strip(),
            cron=(
                cron.strip()
                if cron
                else None
            ),
            interval_minutes=(
                int(interval_minutes)
                if interval_minutes is not None
                else None
            ),
            channels=requested_channels,
            context={
                "client_id": client_id,
                "user_email": user_email,
            },
        )

        # Canonical schedule timing is computed once at creation.
        schedule.set_next_run()

        STORE.add(
            schedule
        )

        return _schedule_output(
            schedule,
            "create",
        )

    except Exception as exc:
        logger.exception(
            "create_schedule failed"
        )

        return (
            f"Error creating schedule: {exc}"
        )


def list_schedules(
    tool_context: ToolContext | None = None,
) -> str:
    """
    List schedules for the current client and, when known,
    the current user.
    """
    client_id = _current_client_id(
        tool_context
    )

    user_email = _current_user_email(
        tool_context
    )

    if not client_id:
        return (
            "Error: no client identified in the current session."
        )

    if not _known_user(user_email):
        return "Error: schedules need a signed-in user."

    schedules = STORE.list(
        user_email=user_email,
        client_id=client_id,
    )

    return json.dumps(
        {
            "status": "success",
            "count": len(schedules),
            "schedules": [
                schedule.model_dump(
                    mode="json"
                )
                for schedule in schedules
            ],
        },
        indent=2,
    )


def update_schedule(
    schedule_id: str,
    name: str | None = None,
    prompt: str | None = None,
    cron: str | None = None,
    interval_minutes: int | None = None,
    active: bool | None = None,
    channels: list[str] | None = None,
    tool_context: ToolContext | None = None,
) -> str:
    """
    Update a schedule while enforcing client and user ownership.

    channels: delivery channel names, e.g. ["internal", "google_chat"].
    """
    client_id = _current_client_id(
        tool_context
    )

    user_email = _current_user_email(
        tool_context
    )

    if not client_id:
        return (
            "Error: no client identified in the current session."
        )

    schedule = STORE.get(
        schedule_id
    )

    if schedule is None:
        return (
            f"Schedule {schedule_id} not found."
        )

    access_error = _ownership_error(
        schedule,
        client_id,
        user_email,
    )

    if access_error:
        return access_error

    updates: dict = {}

    if channels is not None:
        requested_channels = [str(c).strip().lower() for c in channels] or [INTERNAL]
        for channel in requested_channels:
            problem = channel_problem(client_id, channel)
            if problem:
                return f"Error: {problem}"
        updates["channels"] = requested_channels

    if name is not None:
        clean_name = name.strip()

        if not clean_name:
            return (
                "Error: schedule name cannot be empty."
            )

        updates["name"] = clean_name

    if prompt is not None:
        clean_prompt = prompt.strip()

        if not clean_prompt:
            return (
                "Error: schedule prompt cannot be empty."
            )

        updates["prompt"] = clean_prompt

    if (
        cron is not None
        and interval_minutes is not None
    ):
        return (
            "Error: provide only one of cron or interval_minutes."
        )

    frequency_changed = False

    if cron is not None:
        clean_cron = cron.strip()

        updates["cron"] = clean_cron
        updates["interval_minutes"] = None
        frequency_changed = True

    if interval_minutes is not None:
        if int(interval_minutes) < 1:
            return (
                "Error: interval_minutes must be >= 1."
            )

        updates["interval_minutes"] = int(
            interval_minutes
        )
        updates["cron"] = None
        frequency_changed = True

    if active is not None:
        updates["active"] = bool(
            active
        )

    if not updates:
        return (
            "No changes specified."
        )

    try:
        # Reconstruct the proposed schedule first so frequency
        # changes are validated before reaching persistence.
        candidate_data = schedule.model_dump(
            mode="python"
        )

        candidate_data.update(
            updates
        )

        candidate = ScheduleConfig.model_validate(
            candidate_data
        )

        if frequency_changed:
            candidate.set_next_run()

            updates["next_run"] = (
                candidate.next_run.isoformat()
                if candidate.next_run
                else None
            )

            updates["next_run_ts"] = (
                candidate.next_run_ts
            )

        elif (
            active is True
            and schedule.next_run_ts is None
        ):
            candidate.set_next_run()

            updates["next_run"] = (
                candidate.next_run.isoformat()
                if candidate.next_run
                else None
            )

            updates["next_run_ts"] = (
                candidate.next_run_ts
            )

        updated = STORE.update(
            schedule_id,
            updates,
        )

        if updated is None:
            return (
                f"Schedule {schedule_id} not found."
            )

        return _schedule_output(
            updated,
            "update",
        )

    except Exception as exc:
        logger.exception(
            "update_schedule failed"
        )

        return (
            f"Error updating schedule: {exc}"
        )


def delete_schedule(
    schedule_id: str,
    tool_context: ToolContext | None = None,
) -> str:
    """
    Delete an existing schedule while enforcing ownership.
    """
    client_id = _current_client_id(
        tool_context
    )

    user_email = _current_user_email(
        tool_context
    )

    if not client_id:
        return (
            "Error: no client identified in the current session."
        )

    schedule = STORE.get(
        schedule_id
    )

    if schedule is None:
        return (
            f"Schedule {schedule_id} not found."
        )

    access_error = _ownership_error(
        schedule,
        client_id,
        user_email,
    )

    if access_error:
        return access_error

    deleted = STORE.delete(
        schedule_id
    )

    if deleted:
        return (
            f"Schedule '{schedule.name}' "
            f"deleted successfully."
        )

    return (
        f"Schedule {schedule_id} not found."
    )


ROOT_INSTRUCTIONS = """
You are the AstroBot Scheduler specialist.

Manage recurring schedules for the current client.

The application supplies the client and user. Never invent either,
and never pass a client name in a tool call.

SUPPORTED OPERATIONS:
- create_schedule
- list_schedules
- update_schedule
- delete_schedule

FREQUENCY:
- Use exactly one of cron or interval_minutes.
- Cron expressions are UTC.
- If the user supplies a timezone, convert the requested time to UTC
  before creating the cron expression.
- If the user does not supply a timezone, use the project's established
  scheduler default timezone.
- Use interval_minutes for interval-based schedules.

SECURITY:
- Never allow a schedule from another client to be modified or deleted.
- Never expose another user's schedule when the current user identity
  is available.
- Never invent a client, user, or delivery destination.

DELIVERY CHANNELS (pass these names in `channels`):
- "internal": result is stored and visible in AstroBot (default).
- "google_chat": posted to the client's configured Google Chat space.
  Only available when configured for the client; the tool says so if not.
- Email and Sheets delivery are not connected. Say so if asked.
- Never accept a URL or address as a destination.

EXECUTION:
- Do not claim that a schedule has already executed.
- Report delivery only from the recorded delivery status.

SCHEDULING:
- Always return the schedule ID.
- Always preserve the calculated next execution time.
"""

root_agent = Agent(
    name="astrobot_scheduler",
    model="gemini-2.5-flash",
    description=(
        "Manages client-scoped recurring schedules for AstroBot."
    ),
    instruction=ROOT_INSTRUCTIONS,
    tools=[
        create_schedule,
        list_schedules,
        update_schedule,
        delete_schedule,
    ],
)

agent = root_agent