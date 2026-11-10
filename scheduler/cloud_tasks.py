from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from google.api_core.exceptions import AlreadyExists
from google.cloud import tasks_v2


load_dotenv()


logger = logging.getLogger(
    "astrobot-cloud-tasks"
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = (
    os.getenv(
        "CLOUD_TASKS_PROJECT",
        "",
    )
    or os.getenv(
        "GOOGLE_CLOUD_PROJECT",
        "",
    )
    or os.getenv(
        "GCP_PROJECT",
        "",
    )
).strip()


LOCATION = os.getenv(
    "CLOUD_TASKS_LOCATION",
    "us-central1",
).strip()


QUEUE = os.getenv(
    "CLOUD_TASKS_QUEUE",
    "scheduler-queue",
).strip()


SERVICE_ACCOUNT_EMAIL = os.getenv(
    "CLOUD_TASKS_SERVICE_ACCOUNT",
    "",
).strip()


SCHEDULER_RUN_URL = os.getenv(
    "SCHEDULER_RUN_URL",
    "",
).strip()


# ============================================================
# CLIENT
# ============================================================

def get_tasks_client() -> (
    tasks_v2.CloudTasksClient
):
    """
    Create a Cloud Tasks client using ADC.

    No service-account key is loaded.
    """
    return tasks_v2.CloudTasksClient()


# ============================================================
# CONFIGURATION STATUS
# ============================================================

def configuration_status() -> dict[str, Any]:
    """
    Return the current Cloud Tasks configuration.

    A fully operational configuration requires:

    - project
    - location
    - queue
    - service account
    - scheduler Cloud Run URL
    """

    missing: list[str] = []

    if not PROJECT_ID:
        missing.append(
            "CLOUD_TASKS_PROJECT/GOOGLE_CLOUD_PROJECT/GCP_PROJECT"
        )

    if not LOCATION:
        missing.append(
            "CLOUD_TASKS_LOCATION"
        )

    if not QUEUE:
        missing.append(
            "CLOUD_TASKS_QUEUE"
        )

    if not SERVICE_ACCOUNT_EMAIL:
        missing.append(
            "CLOUD_TASKS_SERVICE_ACCOUNT"
        )

    if not SCHEDULER_RUN_URL:
        missing.append(
            "SCHEDULER_RUN_URL"
        )

    if missing:
        return {
            "status": "not_ready",
            "missing": missing,
            "project": PROJECT_ID,
            "location": LOCATION,
            "queue": QUEUE,
            "service_account": (
                SERVICE_ACCOUNT_EMAIL
            ),
            "run_url_configured": bool(
                SCHEDULER_RUN_URL
            ),
        }

    return {
        "status": "ready",
        "project": PROJECT_ID,
        "location": LOCATION,
        "queue": QUEUE,
        "service_account": (
            SERVICE_ACCOUNT_EMAIL
        ),
        "run_url_configured": True,
    }


def require_queue_configuration() -> None:
    """
    Validate all configuration required for creating
    authenticated HTTP tasks.
    """

    status = configuration_status()

    if status["status"] != "ready":
        raise RuntimeError(
            "Cloud Tasks configuration is incomplete: "
            + json.dumps(
                status,
                sort_keys=True,
            )
        )

    if not (
        SCHEDULER_RUN_URL.startswith(
            "https://"
        )
        or SCHEDULER_RUN_URL.startswith(
            "http://"
        )
    ):
        raise RuntimeError(
            "SCHEDULER_RUN_URL must begin with "
            "http:// or https://."
        )


# ============================================================
# QUEUE
# ============================================================

def queue_path() -> str:
    """
    Return the fully qualified Cloud Tasks queue resource name.
    """

    if not PROJECT_ID:
        raise RuntimeError(
            "Cloud Tasks project is not configured."
        )

    client = get_tasks_client()

    return client.queue_path(
        PROJECT_ID,
        LOCATION,
        QUEUE,
    )


def get_queue() -> dict[str, Any]:
    """
    Read the configured queue.
    """

    require_queue_configuration()

    client = get_tasks_client()

    response = client.get_queue(
        request={
            "name": queue_path(),
        }
    )

    return {
        "status": "ok",
        "name": response.name,
        "state": response.state.name,
    }


# ============================================================
# DETERMINISTIC TASK ID
# ============================================================

def deterministic_task_id(
    schedule_id: str,
    next_run_ts: float | int | None,
) -> str:
    """
    Generate a deterministic task ID.

    One schedule execution window maps to exactly one
    deterministic Cloud Tasks task ID.

    Re-running /dispatch therefore produces AlreadyExists
    instead of duplicate tasks.
    """

    material = (
        f"{schedule_id}:{next_run_ts}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        material
    ).hexdigest()

    return (
        "scheduler-"
        + digest[:32]
    )


# ============================================================
# TASK CREATION
# ============================================================

def enqueue_schedule_task(
    schedule_id: str,
    client_id: str,
    next_run_ts: float | int | None,
    user_email: str = "",
) -> dict[str, Any]:
    """
    Create one authenticated Cloud Tasks HTTP task.

    Target:

        {SCHEDULER_RUN_URL}/run/{schedule_id}

    The schedule remains canonical in Firestore.
    """

    require_queue_configuration()

    schedule_id = str(
        schedule_id
    ).strip()

    client_id = str(
        client_id
    ).strip().lower()

    user_email = str(
        user_email or ""
    ).strip().lower()

    if not schedule_id:
        raise ValueError(
            "schedule_id is required."
        )

    if not client_id:
        raise ValueError(
            "client_id is required."
        )

    if next_run_ts is None:
        raise ValueError(
            "next_run_ts is required."
        )

    client = get_tasks_client()

    parent = queue_path()

    task_id = deterministic_task_id(
        schedule_id,
        next_run_ts,
    )

    task_name = client.task_path(
        PROJECT_ID,
        LOCATION,
        QUEUE,
        task_id,
    )

    target_url = (
        SCHEDULER_RUN_URL.rstrip("/")
        + "/run/"
        + schedule_id
    )

    payload = {
        "schedule_id": schedule_id,
        "client_id": client_id,
        "user_email": user_email,
        "next_run_ts": next_run_ts,
    }

    task = tasks_v2.Task(
        name=task_name,
        http_request=tasks_v2.HttpRequest(
            http_method=(
                tasks_v2.HttpMethod.POST
            ),
            url=target_url,
            headers={
                "Content-Type": (
                    "application/json"
                ),
            },
            body=json.dumps(
                payload
            ).encode(
                "utf-8"
            ),
            oidc_token=(
                tasks_v2.OidcToken(
                    service_account_email=(
                        SERVICE_ACCOUNT_EMAIL
                    ),
                    audience=(
                        SCHEDULER_RUN_URL
                    ),
                )
            ),
        ),
    )

    try:
        created = client.create_task(
            request={
                "parent": parent,
                "task": task,
            }
        )

        logger.info(
            (
                "Cloud Task created | "
                "schedule=%s | task=%s"
            ),
            schedule_id,
            created.name,
        )

        return {
            "status": "created",
            "task_name": created.name,
            "schedule_id": schedule_id,
            "client_id": client_id,
            "target_url": target_url,
            "next_run_ts": next_run_ts,
        }

    except AlreadyExists:

        logger.info(
            (
                "Cloud Task already exists | "
                "schedule=%s | task=%s"
            ),
            schedule_id,
            task_name,
        )

        return {
            "status": "already_exists",
            "task_name": task_name,
            "schedule_id": schedule_id,
            "client_id": client_id,
            "target_url": target_url,
            "next_run_ts": next_run_ts,
        }


# ============================================================
# DIAGNOSTIC ENTRY POINT
# ============================================================

def main() -> None:
    """
    Verify configuration and queue connectivity.

    Does not create a task.
    """

    print(
        json.dumps(
            configuration_status(),
            indent=2,
        )
    )

    try:
        print(
            json.dumps(
                get_queue(),
                indent=2,
            )
        )

    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()