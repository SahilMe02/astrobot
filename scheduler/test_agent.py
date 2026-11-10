from __future__ import annotations

import json

from scheduler.agent import (
    create_schedule,
    delete_schedule,
    list_schedules,
    update_schedule,
)


import pytest

import scheduler.agent as _agent

# The real app sets the active client server-side (tenancy.context) right
# before the Scheduler runs. Each _ctx(...) call below stands for one such
# scoped call, so it also sets the active client for that call.
_ACTIVE = {"client": ""}


@pytest.fixture(autouse=True)
def _server_scope(monkeypatch):
    monkeypatch.setattr(_agent, "active_client", lambda: _ACTIVE["client"])
    yield
    _ACTIVE["client"] = ""


class FakeContext:
    def __init__(self, state):
        self.state = state


def _ctx(
    client_id: str = "venetian",
    user_email: str = "test@example.com",
):
    _ACTIVE["client"] = client_id
    return FakeContext(
        {
            "client_id": client_id,
            "user_email": user_email,
        }
    )


def test_create_list_update_delete_schedule(
    tmp_path,
    monkeypatch,
):
    import scheduler.agent as agent_module

    from scheduler.store import ScheduleStore

    monkeypatch.setattr(
        agent_module,
        "STORE",
        ScheduleStore(
            tmp_path
            / "schedules.json"
        ),
    )

    result = create_schedule(
        "Daily email review",
        "Review recent client emails for blockers.",
        cron="0 9 * * 1-5",
        tool_context=_ctx(),
    )

    payload = json.loads(
        result
    )

    assert payload["status"] == "success"

    schedule = payload["schedule"]

    assert schedule["cron"] == "0 9 * * 1-5"
    assert schedule["interval_minutes"] is None
    assert schedule["next_run"] is not None
    assert schedule["next_run_ts"] is not None

    listing = list_schedules(
        tool_context=_ctx()
    )

    listing_payload = json.loads(
        listing
    )

    assert listing_payload["count"] == 1

    schedule_id = schedule["id"]

    updated = update_schedule(
        schedule_id,
        active=False,
        tool_context=_ctx(),
    )

    updated_payload = json.loads(
        updated
    )

    assert (
        updated_payload["schedule"]["active"]
        is False
    )

    deleted = delete_schedule(
        schedule_id,
        tool_context=_ctx(),
    )

    assert (
        "deleted successfully"
        in deleted
    )


def test_cross_client_schedule_isolation(
    tmp_path,
    monkeypatch,
):
    import scheduler.agent as agent_module

    from scheduler.store import ScheduleStore

    monkeypatch.setattr(
        agent_module,
        "STORE",
        ScheduleStore(
            tmp_path
            / "schedules.json"
        ),
    )

    result = create_schedule(
        "Venetian report",
        "Review revenue.",
        interval_minutes=60,
        tool_context=_ctx(),
    )

    schedule_id = json.loads(
        result
    )["schedule"]["id"]

    other = _ctx(
        client_id="npi",
        user_email="test@example.com",
    )

    assert (
        "Access denied"
        in update_schedule(
            schedule_id,
            active=False,
            tool_context=other,
        )
    )

    assert (
        "Access denied"
        in delete_schedule(
            schedule_id,
            tool_context=other,
        )
    )


def test_cross_user_schedule_isolation(
    tmp_path,
    monkeypatch,
):
    import scheduler.agent as agent_module

    from scheduler.store import ScheduleStore

    monkeypatch.setattr(
        agent_module,
        "STORE",
        ScheduleStore(
            tmp_path
            / "schedules.json"
        ),
    )

    result = create_schedule(
        "Venetian report",
        "Review revenue.",
        interval_minutes=60,
        tool_context=_ctx(
            client_id="venetian",
            user_email="owner@example.com",
        ),
    )

    schedule_id = json.loads(
        result
    )["schedule"]["id"]

    other_user = _ctx(
        client_id="venetian",
        user_email="other@example.com",
    )

    assert (
        "Access denied"
        in update_schedule(
            schedule_id,
            active=False,
            tool_context=other_user,
        )
    )

    assert (
        "Access denied"
        in delete_schedule(
            schedule_id,
            tool_context=other_user,
        )
    )


def test_update_frequency_recalculates_next_run(
    tmp_path,
    monkeypatch,
):
    import scheduler.agent as agent_module

    from scheduler.store import ScheduleStore

    monkeypatch.setattr(
        agent_module,
        "STORE",
        ScheduleStore(
            tmp_path
            / "schedules.json"
        ),
    )

    result = create_schedule(
        "Frequency test",
        "Review client status.",
        interval_minutes=60,
        tool_context=_ctx(),
    )

    payload = json.loads(
        result
    )

    schedule_id = payload["schedule"]["id"]
    original_next = payload["schedule"]["next_run_ts"]

    updated = update_schedule(
        schedule_id,
        interval_minutes=120,
        tool_context=_ctx(),
    )

    updated_payload = json.loads(
        updated
    )

    updated_schedule = (
        updated_payload["schedule"]
    )

    assert (
        updated_schedule["interval_minutes"]
        == 120
    )

    assert (
        updated_schedule["cron"]
        is None
    )

    assert (
        updated_schedule["next_run_ts"]
        != original_next
    )


def test_invalid_cron_is_rejected(
    tmp_path,
    monkeypatch,
):
    import scheduler.agent as agent_module

    from scheduler.store import ScheduleStore

    monkeypatch.setattr(
        agent_module,
        "STORE",
        ScheduleStore(
            tmp_path
            / "schedules.json"
        ),
    )

    result = create_schedule(
        "Invalid cron",
        "This should fail.",
        cron="not a cron expression",
        tool_context=_ctx(),
    )

    assert "Error creating schedule" in result