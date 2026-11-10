"""Durable ADK sessions on Vertex AI Agent Engine, with an ownership check
on delete.

ADK's VertexAiSessionService verifies the owner when a session is read,
but not when it is deleted. This subclass reads first (which raises if the
session belongs to someone else) and only then deletes.

Registered as the URI scheme:
  astrobot-agentengine://projects/<p>/locations/<l>/reasoningEngines/<id>
"""

from __future__ import annotations

from google.adk.cli.service_registry import _parse_agent_engine_kwargs, get_service_registry
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

SCHEME = "astrobot-agentengine"


class OwnerCheckedVertexAiSessionService(VertexAiSessionService):
    async def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        session = await self.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            config=GetSessionConfig(num_recent_events=0),
        )  # raises ValueError if the session belongs to another user
        if session is None:
            return
        await super().delete_session(app_name=app_name, user_id=user_id, session_id=session_id)


def _factory(uri: str, **kwargs):
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    params = _parse_agent_engine_kwargs(parsed.netloc + parsed.path, kwargs.get("agents_dir"))
    return OwnerCheckedVertexAiSessionService(**params)


def register() -> None:
    get_service_registry().register_session_service(SCHEME, _factory)
