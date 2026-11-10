"""
AstroBot Economist configuration.
"""

from __future__ import annotations

import os

from tenancy import registry


ECONOMIST_MODEL = os.getenv(
    "ECONOMIST_MODEL",
    "gemini-2.5-flash",
)

# Single source of truth: tenancy/clients.json
SUPPORTED_CLIENTS = set(registry.clients())

MAX_SEARCH_ATTEMPTS = 3

ECONOMIST_CONFIDENCE_THRESHOLD = 0.60