"""Connection registry — tracks which LLM clients the user has connected.

Persistent state at ``~/.config/skein/connections.json``::

    {
      "cursor": {
        "connected_at": "2026-05-08T12:34:56Z",
        "config_paths": [
          "/Users/ameliomar/.cursor/mcp.json",
          "/Users/ameliomar/proj/.cursor/rules/skein.mdc"
        ]
      },
      ...
    }

The registry is the source of truth for ``skein sync`` and ``skein
disconnect``. ``skein up`` syncs only the clients listed here; ``skein
disconnect cursor`` reads the recorded paths and surgically removes the
skein block from each.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import paths as _skein_paths


# Moves to %APPDATA%\skein\connections.json on Windows; unchanged on
# macOS / Linux. See skein/paths.py.
CONNECTIONS_PATH = _skein_paths.connections_path()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load() -> dict[str, dict]:
    if not CONNECTIONS_PATH.exists():
        return {}
    try:
        with open(CONNECTIONS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, dict]) -> None:
    CONNECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONNECTIONS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, CONNECTIONS_PATH)


def get_connected_ids() -> list[str]:
    """Return the list of connected client IDs (sorted)."""
    return sorted(_load().keys())


def is_connected(client_id: str) -> bool:
    return client_id in _load()


def get_connection(client_id: str) -> Optional[dict]:
    return _load().get(client_id)


def list_all() -> dict[str, dict]:
    """Return the full registry."""
    return _load()


def mark_connected(client_id: str, config_paths: list[str]) -> None:
    """Record a successful connect — overwrites any prior entry."""
    data = _load()
    data[client_id] = {
        "connected_at": _now_iso(),
        "config_paths": [str(p) for p in config_paths],
    }
    _save(data)


def mark_disconnected(client_id: str) -> bool:
    """Remove ``client_id`` from the registry. Returns True if it existed."""
    data = _load()
    if client_id not in data:
        return False
    del data[client_id]
    _save(data)
    return True


def get_paths(client_id: str) -> list[str]:
    """Return the config paths recorded for ``client_id`` (empty if none)."""
    entry = get_connection(client_id)
    if not entry:
        return []
    return list(entry.get("config_paths", []))
