from __future__ import annotations

from datetime import datetime, timezone

from scheduler.models import ScheduleConfig
from scheduler.worker import (
    _advance_next_run,
    _build_initial_state,
    _next_schedule_timestamp,
    is_due,
)


def _interval_schedule(
    **kwargs,
):
    base = {
        "name": "Test schedule",
        "prompt": "Check recent emails.",
        "interval_minutes": 60,
        "context": {
            "client_id": "venetian",
            "user_email": "test@example.com",
        },
        "created_at": datetime(
            2026,
            10,
            3,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    }

    base.update(
        kwargs
    )

    schedule = ScheduleConfig(
        **base
    )

    schedule.set_next_run(
        after=schedule.created_at
    )

    return schedule


def _cron_schedule(
    **kwargs,
):
    base = {
        "name": "Cron schedule",
        "prompt": "Review recent emails.",
        "cron": "0 * * * *",
        "context": {
            "client_id": "venetian",
            "user_email": "test@example.com",
        },
        "created_at": datetime(
            2026,
            10,
            3,
            10,
            30,
            tzinfo=timezone.utc,
        ),
    }

    base.update(
        kwargs
    )

    schedule = ScheduleConfig(
        **base
    )

    schedule.set_next_run(
        after=schedule.created_at
    )

    return schedule


def test_initial_state_is_client_locked():
    schedule = _interval_schedule()

    state = _build_initial_state(
        schedule
    )

    assert (
        state["client_id"]
        == "venetian"
    )

    assert (
        state["_current_request_client_id"]
        == "venetian"
    )

    assert (
        state["_current_client_source"]
        == "scheduled"
    )

    assert (
        state["session_context"]["client_locked"]
        is True
    )


def test_interval_schedule_is_not_due_before_first_interval():
    schedule = _interval_schedule()

    now = datetime(
        2026,
        10,
        3,
        10,
        30,
        tzinfo=timezone.utc,
    )

    assert not is_due(
        schedule,
        now,
    )


def test_interval_schedule_is_due_after_first_interval():
    schedule = _interval_schedule()

    now = datetime(
        2026,
        10,
        3,
        11,
        0,
        tzinfo=timezone.utc,
    )

    assert is_due(
        schedule,
        now,
    )


def test_cron_schedule_next_run_is_calculated():
    schedule = _cron_schedule()

    expected = datetime(
        2026,
        10,
        3,
        11,
        0,
        tzinfo=timezone.utc,
    )

    assert (
        schedule.next_run
        == expected
    )

    assert (
        schedule.next_run_ts
        == expected.timestamp()
    )


def test_cron_schedule_is_not_due_before_next_run():
    schedule = _cron_schedule()

    now = datetime(
        2026,
        10,
        3,
        10,
        59,
        tzinfo=timezone.utc,
    )

    assert not is_due(
        schedule,
        now,
    )


def test_cron_schedule_is_due_at_next_run():
    schedule = _cron_schedule()

    now = datetime(
        2026,
        10,
        3,
        11,
        0,
        tzinfo=timezone.utc,
    )

    assert is_due(
        schedule,
        now,
    )


def test_next_schedule_timestamp_supports_interval():
    schedule = _interval_schedule()

    after = datetime(
        2026,
        10,
        3,
        12,
        0,
        tzinfo=timezone.utc,
    )

    expected = datetime(
        2026,
        10,
        3,
        13,
        0,
        tzinfo=timezone.utc,
    ).timestamp()

    assert (
        _next_schedule_timestamp(
            schedule,
            after=after,
        )
        == expected
    )


def test_next_schedule_timestamp_supports_cron():
    schedule = _cron_schedule()

    after = datetime(
        2026,
        10,
        3,
        11,
        0,
        tzinfo=timezone.utc,
    )

    expected = datetime(
        2026,
        10,
        3,
        12,
        0,
        tzinfo=timezone.utc,
    ).timestamp()

    assert (
        _next_schedule_timestamp(
            schedule,
            after=after,
        )
        == expected
    )


def test_advance_next_run_supports_cron():
    schedule = _cron_schedule()

    run_time = datetime(
        2026,
        10,
        3,
        11,
        5,
        tzinfo=timezone.utc,
    )

    updates = _advance_next_run(
        schedule,
        run_time,
    )

    expected_next = datetime(
        2026,
        10,
        3,
        12,
        0,
        tzinfo=timezone.utc,
    )

    assert (
        updates["last_run"]
        == run_time.isoformat()
    )

    assert (
        updates["next_run"]
        == expected_next.isoformat()
    )

    assert (
        updates["next_run_ts"]
        == expected_next.timestamp()
    )