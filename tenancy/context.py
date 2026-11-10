"""Active client for the code currently running.

Set only by server code (orchestrator/delegate.py) right before a
specialist runs. Data-access code reads it from here, never from prompt
text or model-generated tool arguments.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_active_client: ContextVar[str] = ContextVar("astrobot_active_client", default="")


def active_client() -> str:
    """Client for this run ("" = no client, demo data only)."""
    return _active_client.get()


@contextmanager
def client_scope(client_id: str) -> Iterator[None]:
    token = _active_client.set(str(client_id or "").strip().lower())
    try:
        yield
    finally:
        _active_client.reset(token)
