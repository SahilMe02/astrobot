"""Server-owned client registry and access list.

Both files are loaded once and cached. Paths can be overridden with
ASTROBOT_CLIENTS_FILE and ASTROBOT_ACCESS_FILE (used by tests and, later,
replaced by a Firestore loader with the same return shapes).

Every failure fails closed: a missing or invalid file means no clients
and no access.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent


def _load_json(env_var: str, default_name: str) -> dict:
    path = Path(os.getenv(env_var) or _HERE / default_name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.error("Could not load %s; failing closed.", path)
        return {}


@lru_cache(maxsize=1)
def raw_config() -> dict:
    """The whole registry file (clients, demo source, rule profiles)."""
    return _load_json("ASTROBOT_CLIENTS_FILE", "clients.json")


@lru_cache(maxsize=1)
def clients() -> dict[str, dict]:
    """Return {client_id: config} for every registered client."""
    raw = raw_config().get("clients", {})
    return {
        str(cid).strip().lower(): cfg
        for cid, cfg in raw.items()
        if isinstance(cfg, dict) and str(cid).strip()
    }


@lru_cache(maxsize=1)
def _access() -> dict[str, frozenset[str]]:
    raw = _load_json("ASTROBOT_ACCESS_FILE", "access.json").get("users", {})
    return {
        str(user).strip().lower(): frozenset(str(c).strip().lower() for c in allowed)
        for user, allowed in raw.items()
        if isinstance(allowed, list)
    }


def reload() -> None:
    """Drop caches (tests and config refresh)."""
    raw_config.cache_clear()
    clients.cache_clear()
    _access.cache_clear()
    _guest_clients.cache_clear()


def is_client(client_id: str) -> bool:
    return str(client_id or "").strip().lower() in clients()


def display_name(client_id: str) -> str:
    return clients().get(client_id, {}).get("display_name", client_id)


GUEST_PREFIX = "guest-"


@lru_cache(maxsize=1)
def _guest_clients() -> frozenset[str]:
    raw = _load_json("ASTROBOT_ACCESS_FILE", "access.json").get("guests", [])
    return frozenset(str(c).strip().lower() for c in raw) if isinstance(raw, list) else frozenset()


def allowed_clients(user_id: str) -> frozenset[str]:
    """Clients this user may access. Unknown user -> empty set.

    Anonymous visitors ("guest-..." IDs, only ever issued by the web app in
    ASTROBOT_AUTH_MODE=public) get the "guests" list from access.json.
    """
    user = str(user_id or "").strip().lower()
    explicit = _access().get(user)
    if explicit is not None:
        return explicit
    if user.startswith(GUEST_PREFIX) and os.getenv("ASTROBOT_AUTH_MODE", "").strip().lower() == "public":
        return _guest_clients()
    return frozenset()


def has_access(user_id: str, client_id: str) -> bool:
    client_id = str(client_id or "").strip().lower()
    return is_client(client_id) and client_id in allowed_clients(user_id)


def mentioned_clients(text: str) -> set[str]:
    """Every registered client named in free text (by id or alias)."""
    lowered = str(text or "").lower()
    found: set[str] = set()
    for cid, cfg in clients().items():
        names = {cid, *[str(a).lower() for a in cfg.get("aliases", [])]}
        for name in names:
            if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", lowered):
                found.add(cid)
                break
    return found
