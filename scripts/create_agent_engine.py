"""Create (once) the Vertex AI Agent Engine that stores AstroBot sessions.

No agent code is deployed to it; it only holds sessions.

Usage:
  uv run python scripts/create_agent_engine.py
Prints the ASTROBOT_SESSION_SERVICE_URI value to use on Cloud Run.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import vertexai  # noqa: E402

project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
if not project:
    sys.exit("GOOGLE_CLOUD_PROJECT is not set.")

client = vertexai.Client(project=project, location=location)
engine = client.agent_engines.create(config={"display_name": "astrobot-sessions"})
name = engine.api_resource.name
print(f"Created Agent Engine: {name}")
print(f"ASTROBOT_SESSION_SERVICE_URI=astrobot-agentengine://{name}")
