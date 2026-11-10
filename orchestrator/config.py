"""
AstroBot Orchestrator configuration.

The client list lives in tenancy/clients.json (server-owned registry).
Do not duplicate client lists in individual specialist agents.
"""

from typing import Final

from tenancy import registry


ORCHESTRATOR_MODEL: Final[str] = "gemini-2.5-flash"


SPECIALIST_DATA_SCIENCE: Final[str] = "data_science"
SPECIALIST_PERSONA_AGGREGATOR: Final[str] = "persona_aggregator"
SPECIALIST_ECONOMIST: Final[str] = "economist"
SPECIALIST_PROJECT_MANAGER: Final[str] = "project_manager"


SUPPORTED_SPECIALISTS: Final[tuple[str, ...]] = (
    SPECIALIST_DATA_SCIENCE,
    SPECIALIST_PERSONA_AGGREGATOR,
    SPECIALIST_ECONOMIST,
    SPECIALIST_PROJECT_MANAGER,
)


SUPPORTED_CLIENTS: Final[tuple[str, ...]] = tuple(registry.clients())
