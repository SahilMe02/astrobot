"""
Local configuration for the AstroBot Persona Aggregator.
"""

import os

from dotenv import load_dotenv


# Load the project-level .env file.
load_dotenv()


# ============================================================
# Google / model configuration
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    os.getenv(
        "PROJECT_ID",
        "",
    ),
)

LOCATION = os.getenv(
    "GOOGLE_CLOUD_LOCATION",
    os.getenv(
        "LOCATION",
        "us-central1",
    ),
)


# ============================================================
# Persona configuration
# ============================================================

PERSONA_AGENT_MODEL = os.getenv(
    "PERSONA_AGENT_MODEL",
    GEMINI_MODEL,
)

PERSONA_MANAGER_MODEL = os.getenv(
    "PERSONA_MANAGER_MODEL",
    GEMINI_MODEL,
)

MAX_PERSONAS_PER_QUERY = int(
    os.getenv(
        "MAX_PERSONAS_PER_QUERY",
        "10",
    )
)

MAX_REMOTE_CALLS = int(
    os.getenv(
        "MAX_REMOTE_CALLS",
        "3",
    )
)

SEARCH_CACHE_TTL = int(
    os.getenv(
        "SEARCH_CACHE_TTL",
        "3600",
    )
)


# ============================================================
# Client configuration
# ============================================================

DEFAULT_CLIENT = os.getenv(
    "PERSONA_DEFAULT_CLIENT",
    "",
)