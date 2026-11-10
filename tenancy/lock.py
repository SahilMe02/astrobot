"""Session client lock.

A session is bound to at most one client. The first request that names an
authorised client sets the lock; after that the lock never changes. The
lock is re-checked against the access list on every turn, so a forged or
stale lock in session state grants nothing the user is not allowed anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import registry

LOCK_KEY = "client_lock"

_MARKER = re.compile(r"\[\s*client\s*:\s*([^\]]*)\]", re.IGNORECASE)


@dataclass(frozen=True)
class LockDecision:
    client_id: str = ""      # client for this turn ("" = no client)
    newly_locked: bool = False
    block_message: str = ""  # non-empty -> refuse the turn with this text

    @property
    def blocked(self) -> bool:
        return bool(self.block_message)


def explicit_markers(text: str) -> list[str]:
    return [m.strip().lower() for m in _MARKER.findall(text or "")]


def decide(text: str, current_lock: str, user_id: str) -> LockDecision:
    """Decide which client this turn runs under, or refuse it."""
    lock = str(current_lock or "").strip().lower()

    markers = explicit_markers(text)
    unknown = [m for m in markers if not registry.is_client(m)]
    if unknown:
        return LockDecision(block_message=(
            f"'{unknown[0]}' is not a recognised client. "
            "Please check the client name."
        ))

    mentioned = registry.mentioned_clients(text) | set(markers)

    if lock:
        if not registry.has_access(user_id, lock):
            return LockDecision(block_message=(
                "You do not have access to the client this session is "
                "linked to. Please start a new session."
            ))
        others = mentioned - {lock}
        if others:
            other = registry.display_name(sorted(others)[0])
            return LockDecision(block_message=(
                f"This session is locked to {registry.display_name(lock)}. "
                f"Please start a new session to work with {other}."
            ))
        return LockDecision(client_id=lock)

    if len(mentioned) > 1:
        return LockDecision(block_message=(
            "Please work with one client per session. "
            "Which client would you like to start with?"
        ))

    if mentioned:
        client_id = mentioned.pop()
        if not registry.has_access(user_id, client_id):
            return LockDecision(block_message=(
                f"You do not have access to {registry.display_name(client_id)}."
            ))
        return LockDecision(client_id=client_id, newly_locked=True)

    return LockDecision()
