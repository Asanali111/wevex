"""Registry of active Skein projects.

Each call to ``skein up`` registers the project (root directory + scope) so the
daemon can:
  • start a filesystem watcher and incrementally re-ingest on edits
  • re-index everything on daemon restart without the user re-running ``up``

Stored at ``~/.config/skein/projects.json``:

    {
      "projects": [
        {
          "scope": "project:company-brain",
          "root": "/Users/ameliomar/Documents/company-brain",
          "source_root": "company-brain",
          "added_at": "2026-05-08T09:30:00Z",
          "last_ingest": "2026-05-08T10:14:32Z"
        }
      ]
    }

Multiple projects can be active at once (one daemon, many watched dirs).
The registry is the source of truth — the daemon reads it on startup and
re-reads it whenever a project is added/removed via the ``/v1/projects`` API.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("skein.projects")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path.home() / ".config" / "skein" / "projects.json"

_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Project entry
# ---------------------------------------------------------------------------

@dataclass
class ProjectEntry:
    scope: str
    root: str
    source_root: str
    added_at: str = field(default_factory=lambda: _now_iso())
    last_ingest: str | None = None
    watch: bool = True   # whether the daemon should auto-reingest on file changes

    @staticmethod
    def from_dict(d: dict) -> ProjectEntry:
        return ProjectEntry(
            scope=d["scope"],
            root=d["root"],
            source_root=d.get("source_root") or Path(d["root"]).name,
            added_at=d.get("added_at", _now_iso()),
            last_ingest=d.get("last_ingest"),
            watch=bool(d.get("watch", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_raw() -> dict:
    if not REGISTRY_PATH.exists():
        return {"projects": []}
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("projects.json unreadable, starting fresh: %s", e)
        return {"projects": []}


def _save_raw(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    tmp.replace(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_projects() -> list[ProjectEntry]:
    """Return all registered projects, oldest first."""
    with _LOCK:
        data = _load_raw()
        return [ProjectEntry.from_dict(p) for p in data.get("projects", [])]


def get_project(root_or_scope: str) -> ProjectEntry | None:
    """Look up a project by its root path or its scope handle."""
    target = str(Path(root_or_scope).resolve()) if Path(root_or_scope).is_absolute() else root_or_scope
    for p in list_projects():
        if p.root == target or p.scope == root_or_scope:
            return p
    return None


def upsert_project(entry: ProjectEntry) -> ProjectEntry:
    """Add or update a project in the registry. Returns the stored entry."""
    entry.root = str(Path(entry.root).resolve())
    with _LOCK:
        data = _load_raw()
        items = data.setdefault("projects", [])
        for i, p in enumerate(items):
            if p.get("root") == entry.root or p.get("scope") == entry.scope:
                # Preserve original added_at
                entry.added_at = p.get("added_at", entry.added_at)
                items[i] = entry.to_dict()
                _save_raw(data)
                return entry
        items.append(entry.to_dict())
        _save_raw(data)
    return entry


def remove_project(root_or_scope: str) -> bool:
    """Remove a project. Returns True if anything was removed."""
    target_path = str(Path(root_or_scope).resolve()) if Path(root_or_scope).is_absolute() else None
    with _LOCK:
        data = _load_raw()
        items = data.get("projects", [])
        before = len(items)
        items = [
            p for p in items
            if p.get("root") != target_path and p.get("scope") != root_or_scope
        ]
        if len(items) == before:
            return False
        data["projects"] = items
        _save_raw(data)
    return True


def touch_last_ingest(root: str) -> None:
    """Stamp the last_ingest timestamp for a project (called after re-ingest)."""
    target = str(Path(root).resolve())
    with _LOCK:
        data = _load_raw()
        for p in data.get("projects", []):
            if p.get("root") == target:
                p["last_ingest"] = _now_iso()
                _save_raw(data)
                return
