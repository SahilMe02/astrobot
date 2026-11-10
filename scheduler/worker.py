from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import (
    InMemorySessionService,
)
from google.genai import types

from tenancy import registry

from .delivery import deliver

from orchestrator.agent import (
    root_agent as orchestrator_agent,
)

from .models import ScheduleConfig
from .store import ScheduleStore


warnings.filterwarnings(
    "ignore",
    message=(
        r"Warning: there are non-text parts in the response:.*"
    ),
    category=UserWarning,
    module=r"google\.genai\.types",
)


logger = logging.getLogger(
    "astrobot-scheduler-worker"
)


STORE = ScheduleStore()


APP_NAME = (
    "astrobot_scheduler_worker"
)


# ============================================================
# TIME
# ============================================================

def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


# ============================================================
# SCHEDULE CALCULATION
# ============================================================

def _next_schedule_timestamp(
    schedule: ScheduleConfig,
    after: datetime | None = None,
) -> float | None:
    """
    Return the next schedule timestamp.
    """

    next_run = (
        schedule.compute_next_run(
            after=after
        )
    )

    return (
        next_run.timestamp()
        if next_run is not None
        else None
    )


def is_due(
    schedule: ScheduleConfig,
    now: datetime | None = None,
) -> bool:
    """
    Determine whether a schedule is due.

    The persisted next_run_ts is canonical.

    If missing, derive a value from the schedule definition.
    """

    if not schedule.active:
        return False

    current = (
        now
        or _utc_now()
    )

    if current.tzinfo is None:
        current = current.replace(
            tzinfo=timezone.utc
        )

    current = current.astimezone(
        timezone.utc
    )

    current_ts = (
        current.timestamp()
    )

    next_ts = (
        schedule.next_run_ts
    )

    if next_ts is None:
        next_ts = (
            _next_schedule_timestamp(
                schedule,
                after=(
                    schedule.created_at
                    if schedule.last_run
                    is None
                    else schedule.last_run
                ),
            )
        )

    if next_ts is None:
        return False

    return (
        current_ts >= next_ts
    )


# ============================================================
# STATE
# ============================================================

def _build_initial_state(
    schedule: ScheduleConfig,
) -> dict[str, Any]:
    client_id = str(
        schedule.context.get(
            "client_id",
            "",
        )
    ).strip().lower()

    user_email = str(
        schedule.context.get(
            "user_email",
            "scheduler@astrobot.internal",
        )
    ).strip().lower()

    return {
        "client_id": client_id,

        "_current_request_client_id": (
            client_id
        ),

        "_current_client_source": (
            "scheduled"
        ),

        "session_context": {
            "client_id": client_id,
            "client_locked": True,
        },

        "client_locked": True,

        "LOCKED_CLIENT": client_id,

        # Orchestrator lock key. Still re-checked against the access
        # list for the schedule owner on every run.
        "client_lock": client_id,

        "user_email": user_email,
    }


def _build_prompt(
    schedule: ScheduleConfig,
) -> str:
    client_id = str(
        schedule.context.get(
            "client_id",
            "",
        )
    ).strip().lower()

    prompt = (
        schedule.prompt
        .strip()
    )

    if client_id:
        return (
            f"[Client: {client_id}] "
            f"{prompt}"
        )

    return prompt


# ============================================================
# EVENT TEXT
# ============================================================

def _extract_text_from_event(
    event: Any,
) -> str:
    """
    Extract actual text from the final event.

    Tool-call and tool-response parts are ignored.
    """

    content = getattr(
        event,
        "content",
        None,
    )

    if content is None:
        return ""

    parts = getattr(
        content,
        "parts",
        None,
    ) or []

    text_parts: list[str] = []

    for part in parts:
        text = getattr(
            part,
            "text",
            None,
        )

        if (
            isinstance(
                text,
                str,
            )
            and text.strip()
        ):
            text_parts.append(
                text
            )

    return "".join(
        text_parts
    ).strip()


# ============================================================
# RECURRENCE ADVANCEMENT
# ============================================================

def _advance_next_run(
    schedule: ScheduleConfig,
    run_time: datetime,
    scheduled_next_run_ts: float | None = None,
) -> dict[str, Any]:
    """
    Advance recurrence without drift.

    IMPORTANT:

    The next run is based on the scheduled execution slot,
    not on the time the execution happened to finish.

    Example interval:
        scheduled 10:00
        finishes 10:07
        next = 11:00

    NOT:
        next = 11:07

    For delayed schedules, missed slots are skipped until the
    next future slot.
    """

    if run_time.tzinfo is None:
        run_time = run_time.replace(
            tzinfo=timezone.utc
        )

    run_time = run_time.astimezone(
        timezone.utc
    )

    # --------------------------------------------------------
    # Use the original scheduled slot as the recurrence anchor.
    # --------------------------------------------------------

    if (
        scheduled_next_run_ts
        is not None
    ):
        anchor = datetime.fromtimestamp(
            float(
                scheduled_next_run_ts
            ),
            tz=timezone.utc,
        )

    elif schedule.next_run_ts is not None:
        anchor = datetime.fromtimestamp(
            float(
                schedule.next_run_ts
            ),
            tz=timezone.utc,
        )

    else:
        anchor = run_time

    next_run = (
        schedule.compute_next_run(
            after=anchor
        )
    )

    # --------------------------------------------------------
    # If execution was delayed beyond the next slot, keep
    # advancing until the next future slot.
    # --------------------------------------------------------

    safety_counter = 0

    while (
        next_run is not None
        and next_run <= run_time
        and safety_counter < 1000
    ):
        next_run = (
            schedule.compute_next_run(
                after=next_run
            )
        )

        safety_counter += 1

    return {
        "last_run": (
            run_time.isoformat()
        ),

        "next_run": (
            next_run.isoformat()
            if next_run
            else None
        ),

        "next_run_ts": (
            next_run.timestamp()
            if next_run
            else None
        ),
    }


# ============================================================
# EXECUTION
# ============================================================

async def execute_schedule(
    schedule_id: str,
    *,
    force: bool = False,
    expected_next_run_ts: float | int | None = None,
    store: ScheduleStore | None = None,
) -> dict[str, Any]:
    """
    Execute one stored schedule through the real Orchestrator.

    Every scheduled execution gets a fresh ADK session.

    expected_next_run_ts is used to reject stale Cloud Tasks
    that refer to an older execution slot.
    """

    store = (
        store
        or STORE
    )

    # --------------------------------------------------------
    # Load canonical schedule
    # --------------------------------------------------------

    schedule = store.get(
        schedule_id
    )

    if schedule is None:
        return {
            "status": "not_found",
            "schedule_id": schedule_id,
        }

    # --------------------------------------------------------
    # Inactive schedule
    # --------------------------------------------------------

    if not schedule.active:
        return {
            "status": "skipped",
            "reason": "inactive",
            "schedule_id": schedule_id,
        }

    # --------------------------------------------------------
    # Expected task slot validation
    # --------------------------------------------------------

    if (
        expected_next_run_ts
        is not None
        and schedule.next_run_ts
        is not None
    ):

        try:
            expected = float(
                expected_next_run_ts
            )

            canonical = float(
                schedule.next_run_ts
            )

            if abs(
                expected - canonical
            ) > 0.5:
                logger.info(
                    (
                        "Ignoring stale task | "
                        "schedule=%s | expected=%s | canonical=%s"
                    ),
                    schedule_id,
                    expected,
                    canonical,
                )

                return {
                    "status": "stale_task",
                    "schedule_id": schedule_id,
                    "expected_next_run_ts": (
                        expected
                    ),
                    "canonical_next_run_ts": (
                        canonical
                    ),
                }

        except (
            TypeError,
            ValueError,
        ):
            return {
                "status": "error",
                "schedule_id": schedule_id,
                "error": (
                    "invalid_expected_next_run_ts"
                ),
            }

    # --------------------------------------------------------
    # Due check
    # --------------------------------------------------------

    now = _utc_now()

    if (
        not force
        and not is_due(
            schedule,
            now,
        )
    ):
        return {
            "status": "not_due",
            "schedule_id": schedule_id,
            "next_run": (
                schedule.next_run.isoformat()
                if schedule.next_run
                else None
            ),
            "next_run_ts": (
                schedule.next_run_ts
            ),
        }

    # --------------------------------------------------------
    # Client context
    # --------------------------------------------------------

    client_id = str(
        schedule.context.get(
            "client_id",
            "",
        )
    ).strip().lower()

    user_email = str(
        schedule.context.get(
            "user_email",
            "scheduler@astrobot.internal",
        )
    ).strip().lower()

    if not client_id:
        return {
            "status": "error",
            "schedule_id": schedule_id,
            "error": "missing_client_id",
        }

    # Re-check the owner's access at run time: removing a user from the
    # access list stops their scheduled jobs immediately.
    if not registry.has_access(user_email, client_id):
        logger.warning(
            "Scheduled run denied | schedule=%s | owner=%s | client=%s",
            schedule_id, user_email or "unknown", client_id,
        )
        return {
            "status": "access_denied",
            "schedule_id": schedule_id,
            "client_id": client_id,
            "error": "owner_has_no_access_to_client",
        }

    # --------------------------------------------------------
    # Fresh ADK session
    # --------------------------------------------------------

    session_service = (
        InMemorySessionService()
    )

    runner = Runner(
        agent=orchestrator_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Scheduled runs act as the schedule owner, so the owner's
    # current client access is enforced at run time.
    user_id = user_email or "unknown"

    session_id = (
        "sched_"
        + uuid.uuid4().hex[:12]
    )

    initial_state = (
        _build_initial_state(
            schedule
        )
    )

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=_build_prompt(
                    schedule
                )
            )
        ],
    )

    final_text = ""

    try:

        # ----------------------------------------------------
        # Run the Orchestrator
        # ----------------------------------------------------

        async for event in (
            runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message,
            )
        ):

            try:
                final_response = (
                    event.is_final_response()
                )
            except Exception:
                final_response = False

            if not final_response:
                continue

            event_text = (
                _extract_text_from_event(
                    event
                )
            )

            if event_text:
                final_text = event_text

        # ----------------------------------------------------
        # Treat an empty final response as execution failure.
        # This prevents silently advancing a schedule when the
        # Orchestrator produced no actual result.
        # ----------------------------------------------------

        if not final_text:
            logger.error(
                (
                    "Scheduled execution produced "
                    "no final text response | schedule=%s"
                ),
                schedule_id,
            )

            return {
                "status": "error",
                "schedule_id": schedule_id,
                "client_id": client_id,
                "error": (
                    "orchestrator_empty_response"
                ),
            }

        # ----------------------------------------------------
        # Successful execution
        # ----------------------------------------------------

        run_time = _utc_now()

        delivery = deliver(
            client_id,
            schedule.channels,
            title=f"AstroBot scheduled report: {schedule.name}",
            text=final_text,
        )

        schedule_updates = (
            _advance_next_run(
                schedule,
                run_time,
                scheduled_next_run_ts=(
                    schedule.next_run_ts
                ),
            )
        )

        schedule_updates["last_delivery"] = delivery

        updated = store.update(
            schedule_id,
            schedule_updates,
        )

        if updated is None:
            return {
                "status": "error",
                "schedule_id": schedule_id,
                "client_id": client_id,
                "error": (
                    "schedule_update_failed"
                ),
            }

        return {
            "status": "success",
            "schedule_id": schedule_id,
            "client_id": client_id,
            "response": final_text,
            "delivery": delivery,
            "user_email": user_email,
            "last_run": (
                run_time.isoformat()
            ),
            "next_run": (
                updated.next_run.isoformat()
                if updated.next_run
                else None
            ),
            "next_run_ts": (
                updated.next_run_ts
            ),
        }

    except Exception as exc:

        logger.exception(
            (
                "Scheduled execution failed | "
                "schedule=%s"
            ),
            schedule_id,
        )

        # ----------------------------------------------------
        # DO NOT advance recurrence on failure.
        # Cloud Tasks will receive HTTP 500 from /run and retry.
        # ----------------------------------------------------

        return {
            "status": "error",
            "schedule_id": schedule_id,
            "client_id": client_id,
            "error": str(exc),
        }


# ============================================================
# SYNC WRAPPER
# ============================================================

def execute_schedule_sync(
    schedule_id: str,
    *,
    force: bool = False,
    expected_next_run_ts: float | int | None = None,
    store: ScheduleStore | None = None,
) -> dict[str, Any]:
    """
    Synchronous wrapper for:

    - CLI
    - unit tests
    - local scripts

    Cloud Run / FastAPI MUST call execute_schedule()
    directly instead of calling this function.
    """

    return asyncio.run(
        execute_schedule(
            schedule_id,
            force=force,
            expected_next_run_ts=(
                expected_next_run_ts
            ),
            store=store,
        )
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Execute an AstroBot schedule locally "
            "through the Orchestrator."
        )
    )

    parser.add_argument(
        "schedule_id",
        help="Schedule ID to execute.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Execute immediately even when "
            "the schedule is not currently due."
        ),
    )

    args = parser.parse_args()

    result = execute_schedule_sync(
        args.schedule_id,
        force=args.force,
    )

    print(result)


if __name__ == "__main__":
    main()