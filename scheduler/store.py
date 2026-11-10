from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from .models import ScheduleConfig

load_dotenv()


class ScheduleStore:
    """
    Schedule persistence abstraction.

    Local development:
        SCHEDULER_STORAGE=local
        -> schedules.json

    Production:
        SCHEDULER_STORAGE=firestore
        -> Firestore scheduler-db database

    The public ScheduleStore interface remains the same so the
    Scheduler agent does not need to change when the persistence
    backend changes.
    """

    COLLECTION_NAME = "scheduler_schedules"

    def __init__(
        self,
        path: str | Path = "schedules.json",
        backend: str | None = None,
        project_id: str | None = None,
        database: str | None = None,
    ) -> None:
        self.path = Path(path)

        # Test/local stores that explicitly provide another path
        # should remain local even when the project .env is configured
        # for Firestore.
        if backend is None:
            if self.path != Path("schedules.json"):
                backend = "local"
            else:
                backend = os.getenv(
                    "SCHEDULER_STORAGE",
                    "local",
                )

        self.backend = str(
            backend
        ).strip().lower()

        self.project_id = (
            project_id
            or os.getenv(
                "GOOGLE_CLOUD_PROJECT",
                "",
            )
            or os.getenv(
                "GCP_PROJECT",
                "",
            )
        ).strip()

        self.database = (
            database
            or os.getenv(
                "FIRESTORE_DATABASE",
                "scheduler-db",
            )
        ).strip()

        self._lock = threading.Lock()

        if self.backend not in {
            "local",
            "firestore",
        }:
            raise ValueError(
                "SCHEDULER_STORAGE must be "
                "'local' or 'firestore'."
            )

        self._firestore = None

        if self.backend == "local":
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not self.path.exists():
                self.path.write_text(
                    "{}",
                    encoding="utf-8",
                )

        else:
            self._initialize_firestore()

    # ============================================================
    # FIRESTORE INITIALIZATION
    # ============================================================

    def _initialize_firestore(self) -> None:
        """
        Initialize the Firestore client.

        Authentication uses Application Default Credentials.
        """
        from google.cloud import firestore

        self._firestore = firestore.Client(
            project=self.project_id or None,
            database=self.database,
        )

    def _collection(self):
        if self._firestore is None:
            raise RuntimeError(
                "Firestore backend is not initialized."
            )

        return self._firestore.collection(
            self.COLLECTION_NAME
        )

    # ============================================================
    # LOCAL BACKEND
    # ============================================================

    def _load_local(self) -> dict[str, dict]:
        try:
            data = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                data,
                dict,
            ):
                return data

            return {}

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return {}

    def _save_local(
        self,
        data: dict[str, dict],
    ) -> None:
        """
        Atomic local JSON write.
        """
        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            self.path
        )

    # ============================================================
    # SERIALIZATION
    # ============================================================

    @staticmethod
    def _serialize_schedule(
        schedule: ScheduleConfig,
    ) -> dict[str, Any]:
        """
        Convert ScheduleConfig into JSON/Firestore-safe data.
        """
        data = schedule.model_dump(
            mode="python"
        )

        for field in (
            "last_run",
            "next_run",
            "created_at",
        ):
            value = data.get(
                field
            )

            if isinstance(
                value,
                datetime,
            ):
                data[field] = value.isoformat()

        return data

    @staticmethod
    def _deserialize_schedule(
        data: dict[str, Any],
    ) -> ScheduleConfig:
        """
        Convert stored data back into ScheduleConfig.
        """
        return ScheduleConfig.model_validate(
            data
        )

    # ============================================================
    # CREATE
    # ============================================================

    def add(
        self,
        schedule: ScheduleConfig,
    ) -> ScheduleConfig:
        """
        Persist a schedule.
        """
        if schedule.next_run_ts is None:
            schedule.set_next_run()

        data = self._serialize_schedule(
            schedule
        )

        if self.backend == "local":
            with self._lock:
                local_data = self._load_local()

                local_data[
                    schedule.id
                ] = data

                self._save_local(
                    local_data
                )

            return schedule

        self._collection().document(
            schedule.id
        ).set(data)

        return schedule

    # ============================================================
    # GET
    # ============================================================

    def get(
        self,
        schedule_id: str,
    ) -> Optional[ScheduleConfig]:
        if self.backend == "local":
            with self._lock:
                data = (
                    self._load_local()
                    .get(
                        schedule_id
                    )
                )

            if not data:
                return None

            return self._deserialize_schedule(
                data
            )

        document = (
            self._collection()
            .document(schedule_id)
            .get()
        )

        if not document.exists:
            return None

        data = (
            document.to_dict()
            or {}
        )

        return self._deserialize_schedule(
            data
        )

    # ============================================================
    # LIST
    # ============================================================

    def list(
        self,
        user_email: str = "",
        client_id: str = "",
    ) -> list[ScheduleConfig]:
        """
        List schedules.

        Optional client/user filters are applied after retrieval.
        """
        if self.backend == "local":
            with self._lock:
                values = list(
                    self._load_local()
                    .values()
                )

        else:
            values = []

            for document in (
                self._collection().stream()
            ):
                values.append(
                    document.to_dict()
                    or {}
                )

        schedules = [
            self._deserialize_schedule(
                item
            )
            for item in values
        ]

        normalized_user = (
            user_email.strip().lower()
        )

        normalized_client = (
            client_id.strip().lower()
        )

        if normalized_user:
            schedules = [
                schedule
                for schedule in schedules
                if str(
                    schedule.context.get(
                        "user_email",
                        "",
                    )
                ).strip().lower()
                == normalized_user
            ]

        if normalized_client:
            schedules = [
                schedule
                for schedule in schedules
                if str(
                    schedule.context.get(
                        "client_id",
                        "",
                    )
                ).strip().lower()
                == normalized_client
            ]

        return sorted(
            schedules,
            key=lambda schedule: schedule.created_at,
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def update(
        self,
        schedule_id: str,
        updates: dict[str, Any],
    ) -> ScheduleConfig | None:
        """
        Update an existing schedule.

        The final merged object is validated before persistence.
        """
        current = self.get(
            schedule_id
        )

        if current is None:
            return None

        merged = current.model_dump(
            mode="python"
        )

        merged.update(
            updates
        )

        validated = (
            ScheduleConfig.model_validate(
                merged
            )
        )

        data = self._serialize_schedule(
            validated
        )

        if self.backend == "local":
            with self._lock:
                local_data = self._load_local()

                if schedule_id not in local_data:
                    return None

                local_data[
                    schedule_id
                ] = data

                self._save_local(
                    local_data
                )

            return validated

        self._collection().document(
            schedule_id
        ).set(
            data,
            merge=True,
        )

        return validated

    # ============================================================
    # DELETE
    # ============================================================

    def delete(
        self,
        schedule_id: str,
    ) -> bool:
        if self.backend == "local":
            with self._lock:
                data = self._load_local()

                if schedule_id not in data:
                    return False

                del data[
                    schedule_id
                ]

                self._save_local(
                    data
                )

                return True

        document = (
            self._collection()
            .document(schedule_id)
            .get()
        )

        if not document.exists:
            return False

        (
            self._collection()
            .document(schedule_id)
            .delete()
        )

        return True

    # ============================================================
    # MARK RUN
    # ============================================================

    def mark_run(
        self,
        schedule_id: str,
        when: datetime | None = None,
    ) -> ScheduleConfig | None:
        run_time = (
            when
            or datetime.now(
                timezone.utc
            )
        )

        if run_time.tzinfo is None:
            run_time = run_time.replace(
                tzinfo=timezone.utc
            )

        schedule = self.get(
            schedule_id
        )

        if schedule is None:
            return None

        next_run = (
            schedule.compute_next_run(
                after=run_time
            )
        )

        return self.update(
            schedule_id,
            {
                "last_run": run_time,
                "next_run": next_run,
                "next_run_ts": (
                    next_run.timestamp()
                    if next_run
                    else None
                ),
            },
        )

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self) -> dict[str, Any]:
        """
        Verify backend connectivity.
        """
        if self.backend == "local":
            return {
                "status": "ok",
                "backend": "local",
                "path": str(
                    self.path
                ),
            }

        try:
            list(
                self._collection()
                .limit(1)
                .stream()
            )

            return {
                "status": "ok",
                "backend": "firestore",
                "project": self.project_id,
                "database": self.database,
                "collection": self.COLLECTION_NAME,
            }

        except Exception as exc:
            return {
                "status": "error",
                "backend": "firestore",
                "project": self.project_id,
                "database": self.database,
                "collection": self.COLLECTION_NAME,
                "error": str(exc),
            }