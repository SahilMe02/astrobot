from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .cloud_tasks import (
    configuration_status,
    enqueue_schedule_task,
)
from .store import ScheduleStore
from .worker import (
    execute_schedule,
    is_due,
)


logger = logging.getLogger(
    "astrobot-scheduler-service"
)


app = FastAPI(
    title="AstroBot Scheduler",
    version="0.2.0",
)


STORE = ScheduleStore()


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
@app.get("/healthz")
def health() -> dict[str, Any]:
    """
    Health endpoint for:

    - local verification
    - Cloud Run health checks
    - operational diagnostics
    """

    cloud_tasks = configuration_status()

    store_health = STORE.health_check()

    overall_ok = (
        store_health.get("status") == "ok"
        and cloud_tasks.get("status")
        in {
            "ready",
            "not_ready",
        }
    )

    return {
        "status": (
            "ok"
            if overall_ok
            else "degraded"
        ),
        "service": "astrobot-scheduler",
        "store": store_health,
        "cloud_tasks": cloud_tasks,
    }


# ============================================================
# DISPATCH
# ============================================================

@app.post("/dispatch")
def dispatch_due_schedules() -> dict[str, Any]:
    """
    Find due schedules in Firestore and enqueue one deterministic
    Cloud Task per due execution.

    Cloud Scheduler is expected to call this endpoint periodically.

    The schedule itself remains the source of truth. Cloud Tasks
    merely triggers execution.
    """

    cloud_tasks_status = (
        configuration_status()
    )

    if cloud_tasks_status.get("status") != "ready":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "cloud_tasks_not_ready",
                "configuration": (
                    cloud_tasks_status
                ),
            },
        )

    try:
        now = _utc_now()

        schedules = STORE.list()

        due_schedules: list[Any] = []

        for schedule in schedules:

            if not schedule.active:
                continue

            # ------------------------------------------------
            # Ensure every active schedule has a canonical
            # persisted next_run_ts.
            # ------------------------------------------------

            if schedule.next_run_ts is None:

                schedule.set_next_run(
                    after=(
                        schedule.last_run
                        or schedule.created_at
                    )
                )

                STORE.update(
                    schedule.id,
                    {
                        "next_run": (
                            schedule.next_run.isoformat()
                            if schedule.next_run
                            else None
                        ),
                        "next_run_ts": (
                            schedule.next_run_ts
                        ),
                    },
                )

            if is_due(
                schedule,
                now,
            ):
                due_schedules.append(
                    schedule
                )

        dispatched: list[
            dict[str, Any]
        ] = []

        for schedule in due_schedules:

            client_id = str(
                schedule.context.get(
                    "client_id",
                    "",
                )
            ).strip().lower()

            user_email = str(
                schedule.context.get(
                    "user_email",
                    "",
                )
            ).strip().lower()

            if not client_id:
                logger.warning(
                    (
                        "Skipping due schedule with "
                        "missing client_id: %s"
                    ),
                    schedule.id,
                )
                continue

            if (
                schedule.next_run_ts
                is None
            ):
                logger.warning(
                    (
                        "Skipping due schedule with "
                        "missing next_run_ts: %s"
                    ),
                    schedule.id,
                )
                continue

            result = (
                enqueue_schedule_task(
                    schedule_id=schedule.id,
                    client_id=client_id,
                    next_run_ts=(
                        schedule.next_run_ts
                    ),
                    user_email=user_email,
                )
            )

            dispatched.append(
                {
                    "schedule_id": schedule.id,
                    "name": schedule.name,
                    "client_id": client_id,
                    "next_run_ts": (
                        schedule.next_run_ts
                    ),
                    "task": result,
                }
            )

        return {
            "status": "success",
            "checked_at": now.isoformat(),
            "total_schedules": len(
                schedules
            ),
            "due_schedules": len(
                due_schedules
            ),
            "dispatched": dispatched,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to dispatch due schedules."
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "dispatch_failed",
                "message": str(exc),
            },
        ) from exc


# ============================================================
# RUN
# ============================================================

@app.post("/run/{schedule_id}")
async def run_schedule(
    schedule_id: str,
    request: Request,
) -> dict[str, Any]:
    """
    Execute one schedule asynchronously.

    IMPORTANT:
    This endpoint MUST remain async.

    The worker itself is async and must NOT be wrapped with
    asyncio.run() while FastAPI is already running an event loop.
    """

    schedule_id = str(
        schedule_id
    ).strip()

    if not schedule_id:
        raise HTTPException(
            status_code=400,
            detail="schedule_id is required.",
        )

    # --------------------------------------------------------
    # Read Cloud Tasks payload when available.
    # Payload is advisory metadata only.
    # The canonical schedule remains Firestore.
    # --------------------------------------------------------

    payload: dict[str, Any] = {}

    try:
        raw_payload = await request.json()

        if isinstance(
            raw_payload,
            dict,
        ):
            payload = raw_payload

    except Exception:
        # Empty/non-JSON task body is allowed.
        payload = {}

    payload_schedule_id = str(
        payload.get(
            "schedule_id",
            "",
        )
        or ""
    ).strip()

    if (
        payload_schedule_id
        and payload_schedule_id
        != schedule_id
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "schedule_id_mismatch",
                "path_schedule_id": schedule_id,
                "payload_schedule_id": (
                    payload_schedule_id
                ),
            },
        )

    # --------------------------------------------------------
    # Canonical schedule lookup before execution.
    # --------------------------------------------------------

    schedule = STORE.get(
        schedule_id
    )

    if schedule is None:
        # The schedule may have been deleted after the
        # Cloud Task was created.
        #
        # Return 200 so Cloud Tasks acknowledges the task
        # rather than retrying forever.
        return {
            "status": "not_found",
            "schedule_id": schedule_id,
        }

    canonical_client_id = str(
        schedule.context.get(
            "client_id",
            "",
        )
    ).strip().lower()

    payload_client_id = str(
        payload.get(
            "client_id",
            "",
        )
        or ""
    ).strip().lower()

    if (
        payload_client_id
        and canonical_client_id
        and payload_client_id
        != canonical_client_id
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "client_id_mismatch",
                "schedule_id": schedule_id,
                "canonical_client_id": (
                    canonical_client_id
                ),
                "payload_client_id": (
                    payload_client_id
                ),
            },
        )

    expected_next_run_ts = payload.get(
        "next_run_ts"
    )

    # --------------------------------------------------------
    # Execute through the native async worker.
    # --------------------------------------------------------

    try:
        result = await execute_schedule(
            schedule_id,
            expected_next_run_ts=(
                expected_next_run_ts
            ),
        )

    except Exception as exc:
        logger.exception(
            "Scheduled execution failed: %s",
            schedule_id,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "scheduled_execution_failed",
                "schedule_id": schedule_id,
                "message": str(exc),
            },
        ) from exc

    status = result.get(
        "status"
    )

    # --------------------------------------------------------
    # Safe acknowledgements
    # --------------------------------------------------------

    if status in {
        "not_found",
        "skipped",
        "not_due",
        "stale_task",
    }:
        return result

    # --------------------------------------------------------
    # Actual execution error
    # --------------------------------------------------------

    if status == "error":
        raise HTTPException(
            status_code=500,
            detail=result,
        )

    # --------------------------------------------------------
    # Successful execution
    # --------------------------------------------------------

    if status == "success":
        return result

    # --------------------------------------------------------
    # Unknown result contract
    # --------------------------------------------------------

    raise HTTPException(
        status_code=500,
        detail={
            "error": "unknown_scheduler_result",
            "result": result,
        },
    )