"""Regression: a second persona request in the same session must evaluate the
NEW campaign. The full Persona pipeline runs (root -> workflow -> 9 persona
analysts -> manager) with a fake model that echoes what each agent received."""

import asyncio

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.models import LlmResponse
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from orchestrator.delegate import run_specialist
from persona_aggregator.agent import root_agent as persona


def _text(content):
    return " ".join(p.text for p in (content.parts or []) if getattr(p, "text", None))


async def _echo_model(self, llm_request, stream=False):
    contents = llm_request.contents or []
    last = contents[-1].parts[0] if contents else None
    tools = [
        f.name
        for t in (llm_request.config.tools or [])
        for f in (getattr(t, "function_declarations", None) or [])
    ]
    response = getattr(last, "function_response", None)
    if "run_persona_workflow" in tools and response is None:
        part = types.Part.from_function_call(
            name="run_persona_workflow", args={"request": _text(contents[-1])}
        )
    elif response is not None:
        part = types.Part.from_text(text=str(response.response))
    else:
        part = types.Part.from_text(
            text="saw: " + " | ".join(_text(c) for c in contents if c.role == "user")
        )
    yield LlmResponse(content=types.Content(role="model", parts=[part]))


class _Parent(BaseAgent):
    message: str = ""

    async def _run_async_impl(self, ctx):
        tool_context = ToolContext(ctx)
        text = await run_specialist(
            persona, self.message, tool_context, specialist="persona_aggregator",
            base_state={"client_id": "venetian", "client_lock": "venetian"},
        )
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    content=types.Content(role="model", parts=[types.Part(text=text)]),
                    actions=tool_context.actions)


def test_second_request_evaluates_new_campaign(monkeypatch):
    monkeypatch.setattr(Gemini, "generate_content_async", _echo_model)

    async def run():
        service = InMemorySessionService()
        session = await service.create_session(app_name="t", user_id="u")
        outputs = []
        for message in ("Evaluate: SPA DAY OFFER", "Evaluate: SHOW TICKETS OFFER"):
            runner = Runner(app_name="t", agent=_Parent(name="p", message=message),
                            session_service=service)
            final = ""
            async for event in runner.run_async(
                user_id="u", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="go")]),
            ):
                if event.content and event.content.parts and event.content.parts[0].text:
                    final = event.content.parts[0].text
            outputs.append(final)
        return outputs

    first, second = asyncio.run(run())
    assert "SPA DAY OFFER" in first
    assert "SHOW TICKETS OFFER" in second
    assert "SPA DAY OFFER" not in second
    assert "cached_result" not in second
