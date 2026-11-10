"""Orchestrator isolation: client lock callback and scoped specialist runs."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from orchestrator.agent import before_agent_callback
from orchestrator.delegate import private_state_key, run_specialist
from tenancy import registry


@pytest.fixture(autouse=True)
def dev_files(tmp_path, monkeypatch):
    clients = {"clients": {
        "npi": {"display_name": "NPI", "aliases": ["npi"]},
        "venetian": {"display_name": "The Venetian", "aliases": ["venetian"]},
    }}
    (tmp_path / "c.json").write_text(json.dumps(clients))
    (tmp_path / "a.json").write_text(json.dumps({"users": {"analyst": ["npi", "venetian"]}}))
    monkeypatch.setenv("ASTROBOT_CLIENTS_FILE", str(tmp_path / "c.json"))
    monkeypatch.setenv("ASTROBOT_ACCESS_FILE", str(tmp_path / "a.json"))
    registry.reload()
    yield
    registry.reload()


# ------------------------------------------------------------
# Callback
# ------------------------------------------------------------

def _ctx(text, state, user="analyst"):
    content = types.Content(role="user", parts=[types.Part(text=text)])
    return SimpleNamespace(state=state, user_id=user, user_content=content)


def test_callback_locks_then_blocks_switch():
    state = {}
    assert before_agent_callback(_ctx("[Client: npi] revenue", state)) is None
    assert state["client_lock"] == "npi" and state["client_id"] == "npi"

    refused = before_agent_callback(_ctx("Venetian revenue", state))
    assert refused is not None and "locked to NPI" in refused.parts[0].text
    assert state["client_lock"] == "npi"


def test_callback_restores_client_id_from_lock():
    state = {"client_lock": "npi", "client_id": "venetian"}  # tampered client_id
    assert before_agent_callback(_ctx("show revenue", state)) is None
    assert state["client_id"] == "npi"


def test_callback_marks_client_only_when_named_this_turn():
    state = {}
    before_agent_callback(_ctx("[Client: npi] flight prices?", state))
    assert state["_current_request_client_id"] == "npi"
    before_agent_callback(_ctx("Generic flight price trends?", state))
    assert state["_current_request_client_id"] == ""
    assert state["client_id"] == "npi"


# ------------------------------------------------------------
# Scoped specialist run
# ------------------------------------------------------------

class ProbeAgent(BaseAgent):
    """Reports the state it sees and tries to write keys."""

    writes: dict = {}

    async def _run_async_impl(self, ctx):
        seen = sorted(ctx.session.state.keys())
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text=json.dumps(seen))]),
            actions=EventActions(state_delta=dict(self.writes)),
        )


class ParentAgent(BaseAgent):
    """Calls run_specialist the way an Orchestrator tool does."""

    child: BaseAgent
    specialist: str = "probe"

    async def _run_async_impl(self, ctx):
        tool_context = ToolContext(ctx)
        text = await run_specialist(
            self.child, "do work", tool_context,
            specialist=self.specialist,
            base_state={"client_id": "npi", "client_lock": "npi"},
        )
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            actions=tool_context.actions,
        )


async def _run(parent, initial_state):
    service = InMemorySessionService()
    session = await service.create_session(app_name="t", user_id="analyst", state=initial_state)
    runner = Runner(app_name="t", agent=parent, session_service=service)
    final = ""
    async for event in runner.run_async(
        user_id="analyst", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    ):
        if event.content and event.content.parts:
            final = event.content.parts[0].text
    return final, (await service.get_session(app_name="t", user_id="analyst",
                                             session_id=session.id)).state


def test_specialist_sees_only_base_and_own_state():
    probe = ProbeAgent(name="probe")
    parent = ParentAgent(name="parent", child=probe)
    seen, _ = asyncio.run(_run(parent, {
        "client_lock": "npi",
        "last_email_threads": "secret emails",
        private_state_key("other"): {"rows": [1, 2]},
        private_state_key("probe"): {"my_earlier_result": "ok"},
    }))
    assert json.loads(seen) == ["client_id", "client_lock", "my_earlier_result"]


def test_specialist_cannot_change_client_and_writes_stay_private():
    probe = ProbeAgent(name="probe", writes={
        "client_id": "venetian", "client_lock": "venetian", "app:shared": 1, "my_result": 42,
    })
    parent = ParentAgent(name="parent", child=probe)
    _, state = asyncio.run(_run(parent, {"client_lock": "npi", "client_id": "npi"}))
    assert state["client_lock"] == "npi" and state["client_id"] == "npi"
    assert "my_result" not in state
    assert state[private_state_key("probe")] == {"my_result": 42}
    assert "app:shared" not in state[private_state_key("probe")]


def test_persona_request_carries_current_message():
    from orchestrator.agent import _with_current_message

    ctx = SimpleNamespace(state={"_current_user_request": "[Client: npi] Evaluate: '50% off groceries'"})
    stale = "Evaluate: 'Book a weekend suite and get a free spa day'"
    routed = _with_current_message(stale, ctx)
    assert "50% off groceries" in routed and "authoritative" in routed
    assert "[Client:" not in routed.split("authoritative")[1]
    same = "Evaluate: '50% off groceries'"
    assert _with_current_message(same, ctx) == same


def test_routing_rules_send_holiday_questions_to_economist():
    from orchestrator.agent import ROOT_INSTRUCTIONS

    assert "Public holidays and holiday calendars" in ROOT_INSTRUCTIONS
    assert "never to\ncall_scheduler" in ROOT_INSTRUCTIONS


def test_kpi_explanations_measure_internal_data_first():
    from orchestrator.agent import ROOT_INSTRUCTIONS

    assert "EXPLAINING A CLIENT KPI MOVEMENT" in ROOT_INSTRUCTIONS
    assert "Data Science FIRST" in ROOT_INSTRUCTIONS
