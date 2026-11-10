from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from croniter import croniter
from pydantic import BaseModel, Field, model_validator


class ScheduleConfig(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    name: str
    prompt: str

    target_agent: str = "orchestrator"

    cron: Optional[str] = None
    interval_minutes: Optional[int] = None

    channels: list[str] = Field(
        default_factory=lambda: ["internal"]
    )

    active: bool = True

    context: dict[str, Any] = Field(
        default_factory=dict
    )

    last_run: Optional[datetime] = None

    # Per-channel delivery status of the most recent run.
    last_delivery: list[dict[str, Any]] = Field(
        default_factory=list
    )

    next_run: Optional[datetime] = None

    next_run_ts: Optional[float] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_frequency(self) -> "ScheduleConfig":
        # Exactly one recurrence mechanism is required.
        has_cron = bool(
            self.cron
            and self.cron.strip()
        )

        has_interval = (
            self.interval_minutes is not None
        )

        if has_cron == has_interval:
            raise ValueError(
                "Provide exactly one of cron or interval_minutes."
            )

        if self.interval_minutes is not None:
            if self.interval_minutes < 1:
                raise ValueError(
                    "interval_minutes must be >= 1."
                )

        if has_cron:
            self.cron = self.cron.strip()

            if not croniter.is_valid(self.cron):
                raise ValueError(
                    f"Invalid cron expression: {self.cron}"
                )

        # Always keep timestamps timezone-aware.
        if (
            self.created_at.tzinfo is None
        ):
            self.created_at = self.created_at.replace(
                tzinfo=timezone.utc
            )

        if (
            self.last_run is not None
            and self.last_run.tzinfo is None
        ):
            self.last_run = self.last_run.replace(
                tzinfo=timezone.utc
            )

        if (
            self.next_run is not None
            and self.next_run.tzinfo is None
        ):
            self.next_run = self.next_run.replace(
                tzinfo=timezone.utc
            )

        return self

    def compute_next_run(
        self,
        after: datetime | None = None,
    ) -> datetime | None:
        """
        Compute the next execution time after the supplied timestamp.

        Cron schedules:
            Uses croniter, matching the reference implementation.

        Interval schedules:
            Adds interval_minutes to the supplied timestamp.

        The returned datetime is always timezone-aware UTC.
        """
        base = after or datetime.now(timezone.utc)

        if base.tzinfo is None:
            base = base.replace(
                tzinfo=timezone.utc
            )

        base = base.astimezone(timezone.utc)

        if self.cron:
            next_dt = croniter(
                self.cron,
                base,
            ).get_next(datetime)

            if next_dt.tzinfo is None:
                next_dt = next_dt.replace(
                    tzinfo=timezone.utc
                )

            return next_dt.astimezone(
                timezone.utc
            )

        if self.interval_minutes is not None:
            return (
                base
                + timedelta(
                    minutes=self.interval_minutes
                )
            )

        return None

    def set_next_run(
        self,
        after: datetime | None = None,
    ) -> datetime | None:
        """
        Compute and store next_run and next_run_ts.
        """
        next_run = self.compute_next_run(
            after=after
        )

        self.next_run = next_run

        self.next_run_ts = (
            next_run.timestamp()
            if next_run is not None
            else None
        )

        return next_run