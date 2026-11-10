"""
Internal result contract for AstroBot specialist calls.

This module defines the small, shared envelope used by the Orchestrator to
record specialist results in session state. It does not change the text that
an internal specialist returns to the Orchestrator or what the user sees.

The contract is intentionally minimal so new specialists (including the future
Project Manager) can use the same boundary without requiring a redesign.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ResultStatus = Literal["success", "error", "unavailable"]


@dataclass(frozen=True)
class SpecialistResult:
    """Normalized internal result for one specialist invocation."""

    specialist: str
    status: ResultStatus
    output: str = ""
    error: str = ""
    client_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary for session state."""
        return asdict(self)

    def to_json(self) -> str:
        """Return a machine-readable representation for internal use only."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def make_success_result(
    specialist: str,
    output: str,
    *,
    client_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> SpecialistResult:
    """Create a successful specialist result."""
    return SpecialistResult(
        specialist=specialist,
        status="success",
        output=str(output),
        client_id=str(client_id or ""),
        metadata=dict(metadata or {}),
    )


def make_error_result(
    specialist: str,
    error: str,
    *,
    client_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> SpecialistResult:
    """Create a failed specialist result."""
    return SpecialistResult(
        specialist=specialist,
        status="error",
        error=str(error),
        client_id=str(client_id or ""),
        metadata=dict(metadata or {}),
    )


def make_unavailable_result(
    specialist: str,
    output: str,
    *,
    client_id: str = "",
    reason: str = "missing_required_configuration",
    metadata: dict[str, Any] | None = None,
) -> SpecialistResult:
    """Create a result for a specialist that is valid but not currently usable."""
    return SpecialistResult(
        specialist=specialist,
        status="unavailable",
        output=str(output),
        error=str(reason),
        client_id=str(client_id or ""),
        metadata=dict(metadata or {}),
    )


def store_specialist_result(
    state: Any,
    result: SpecialistResult,
) -> None:
    """Persist the normalized result without changing the user-facing output."""
    existing = state.get("specialist_results", {})

    if not isinstance(existing, dict):
        existing = {}

    existing[result.specialist] = result.to_dict()

    state["specialist_results"] = existing
    state["last_specialist_result"] = result.to_dict()
    state["last_specialist"] = result.specialist
