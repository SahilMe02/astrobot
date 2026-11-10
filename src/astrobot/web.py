"""Orchestrator-only Google ADK Web UI for AstroBot."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.agent_loader import AgentLoader

from astrobot import sessions
from astrobot.identity import IdentityMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.is_file():
    load_dotenv(dotenv_path=ENV_FILE, override=False)

if not (PROJECT_ROOT / "orchestrator" / "agent.py").is_file():
    raise RuntimeError(
        f"AstroBot project root was not found at {PROJECT_ROOT}"
    )


class OrchestratorOnlyAgentLoader(AgentLoader):
    """Expose only the public Orchestrator app through ADK."""

    PUBLIC_APP_NAME = "orchestrator"

    def list_agents(self) -> list[str]:
        """Show only the Orchestrator in the ADK app selector."""
        return [self.PUBLIC_APP_NAME]

    def load_agent(self, agent_name: str):
        """Reject direct app access to specialist agents."""
        if agent_name != self.PUBLIC_APP_NAME:
            raise HTTPException(
                status_code=404,
                detail=(
                    "This app is not publicly available. "
                    "Please use the AstroBot Orchestrator."
                ),
            )

        return super().load_agent(agent_name)


PORT = int(os.environ.get("PORT", "8080"))

# Durable storage (step 10). Unset = in-memory (local development only).
#   ASTROBOT_SESSION_SERVICE_URI  e.g. astrobot-agentengine://projects/P/locations/L/reasoningEngines/ID
#   ASTROBOT_ARTIFACT_SERVICE_URI e.g. gs://<bucket-name>
SESSION_SERVICE_URI = os.getenv("ASTROBOT_SESSION_SERVICE_URI") or None
ARTIFACT_SERVICE_URI = os.getenv("ASTROBOT_ARTIFACT_SERVICE_URI") or None

# Browser origins allowed to make state-changing requests (ADK's CSRF check).
# Needed when the UI is reached through `gcloud run services proxy`, e.g.
#   ASTROBOT_ALLOWED_ORIGINS=http://127.0.0.1:8081,http://localhost:8081
# Never use "*".
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ASTROBOT_ALLOWED_ORIGINS", "").split(",")
    if o.strip() and o.strip() != "*"
] or None

sessions.register()  # astrobot-agentengine:// (owner-checked deletes)

adk_app = get_fast_api_app(
    agents_dir=str(PROJECT_ROOT),
    agent_loader=OrchestratorOnlyAgentLoader(str(PROJECT_ROOT)),
    web=True,
    use_local_storage=False,
    session_service_uri=SESSION_SERVICE_URI,
    artifact_service_uri=ARTIFACT_SERVICE_URI,
    allow_origins=ALLOWED_ORIGINS,
    host="0.0.0.0",
    port=PORT,
)

# Every request runs as the verified user (see identity.py).
app = IdentityMiddleware(adk_app)
