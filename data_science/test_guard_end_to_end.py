"""End-to-end: a model inside the real BigQuery agent tries to read another
client's table through execute_sql. The run goes through the same
run_specialist path the Orchestrator uses, so the active client comes from
the server-side scope. BigQuery must never be called."""

import asyncio
import json

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.models import BaseLlm, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from orchestrator.delegate import run_specialist
from tenancy import registry


class ScriptedLlm(BaseLlm):
    """Calls execute_sql once with the given SQL, then reports the tool result."""

    sql: str = ""

    async def generate_content_async(self, llm_request, stream=False):
        last = llm_request.contents[-1].parts[0] if llm_request.contents else None
        response = getattr(last, "function_response", None)
        if response is None:
            part = types.Part.from_function_call(
                name="execute_sql",
                args={"project_id": "proj-1", "query": self.sql},
            )
        else:
            part = types.Part.from_text(text=json.dumps(response.response))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class Parent(BaseAgent):
    child: BaseAgent
    client: str = "npi"

    async def _run_async_impl(self, ctx):
        tool_context = ToolContext(ctx)
        text = await run_specialist(
            self.child, "query data", tool_context, specialist="data_science",
            base_state={"client_id": self.client, "client_lock": self.client,
                        "database_settings": {"client_id": self.client, "tables": []}},
        )
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    content=types.Content(role="model", parts=[types.Part(text=text)]))


def _run(sql, monkeypatch, client="npi"):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    registry.reload()

    from google.adk.tools.bigquery import query_tool
    calls = []

    def execute_sql(project_id, query, credentials, settings, tool_context, dry_run=False):
        """Stand-in for the real BigQuery call."""
        calls.append(query)
        return {"status": "SUCCESS", "rows": []}

    monkeypatch.setattr(query_tool, "execute_sql", execute_sql)

    from data_science.sub_agents.bigquery.agent import bigquery_agent
    child = bigquery_agent.model_copy(update={"model": ScriptedLlm(model="scripted", sql=sql)})

    async def go():
        service = InMemorySessionService()
        session = await service.create_session(app_name="t", user_id="analyst")
        runner = Runner(app_name="t", agent=Parent(name="p", child=child, client=client),
                        session_service=service)
        out = ""
        async for event in runner.run_async(
            user_id="analyst", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="go")]),
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                out = event.content.parts[0].text
        return out

    return asyncio.run(go()), calls


def test_other_client_table_is_blocked_before_bigquery(monkeypatch):
    out, calls = _run("SELECT * FROM `proj-1.astrobot_venetian.performance`", monkeypatch)
    assert "blocked by AstroBot data policy" in out
    assert calls == []


def test_own_table_reaches_execution(monkeypatch):
    out, calls = _run("SELECT SUM(Cost) FROM `proj-1.astrobot_npi.performance`", monkeypatch)
    assert "blocked" not in out
    assert len(calls) == 1
