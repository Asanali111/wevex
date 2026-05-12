"""Append-only JSONL event log for live introspection.

R-02: lets the user run ``skein tail`` to watch in real time what the daemon is
doing — every ``recall``, ``remember``, ``supersede``, ``claim_lease``,
``release_lease``, ``note_decision`` from the MCP layer, plus hook injection
events from ``hooks.py``.

Design:

- Single JSONL file at ``$SKEIN_EVENTS_PATH`` (env override) or
  ``~/.config/skein/events.jsonl`` (default). Sits next to ``skein.db``.
- Append-only writes via ``open(path, "a", buffering=1)``. Each line < PIPE_BUF
  (4 KiB) so POSIX guarantees atomic-append across processes — safe to write
  from the daemon AND from hook subprocesses.
- Rotation: when the file exceeds ``MAX_BYTES`` (default 5 MB) on next write,
  rename to ``events.jsonl.1`` and start fresh. Two rotations max
  (``events.jsonl``, ``.1``, ``.2`` — older rotations are deleted).
- Logger is a thin wrapper. NO embedded log levels, NO structured filtering
  here — the tail tool does that with grep/awk.

Failure mode: if the events file can't be opened, ``log()`` is a no-op. We
never break a recall/remember call because of an event log error.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("skein.events")

# 5 MiB. At ~150 bytes/event this holds roughly 33k events before rotation.
MAX_BYTES = 5 * 1024 * 1024
KEEP_ROTATIONS = 2


def default_path() -> Path:
    env = os.environ.get("SKEIN_EVENTS_PATH")
    if env:
        return Path(env)
    return Path.home() / ".config" / "skein" / "events.jsonl"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class EventLogger:
    """Append-only JSONL event logger.

    Thread/process-safe at the line level — relies on POSIX O_APPEND atomicity.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = path or default_path()
        # Caller may construct an EventLogger before the path exists. Parent
        # dir is created lazily on first write so a misconfigured path doesn't
        # crash daemon boot.
        self._parent_ready = False

    # ---- public ----

    def log(self, event: str, scope: Optional[str] = None, **fields: Any) -> None:
        """Append one event. Never raises."""
        try:
            self._maybe_rotate()
            record: Dict[str, Any] = {
                "ts": _now_iso(),
                "event": event,
            }
            if scope is not None:
                record["scope"] = scope
            if fields:
                record["details"] = fields
            line = json.dumps(record, separators=(",", ":"))
            self._append(line)
        except Exception:
            # Logging must never break the caller. Swallow.
            logger.debug("events.log failed", exc_info=True)

    # ---- internals ----

    def _ensure_parent(self) -> None:
        if not self._parent_ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._parent_ready = True

    def _append(self, line: str) -> None:
        self._ensure_parent()
        # Open in append mode for every write. Slow vs. holding a handle, but
        # safe across forks/threads and survives external truncation.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _maybe_rotate(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < MAX_BYTES:
            return
        # Rotate: drop the oldest, shift the rest.
        oldest = self.path.with_suffix(self.path.suffix + f".{KEEP_ROTATIONS}")
        try:
            oldest.unlink()
        except FileNotFoundError:
            pass
        # Walk from N-1 down to 1, shifting each up one.
        for n in range(KEEP_ROTATIONS - 1, 0, -1):
            src = self.path.with_suffix(self.path.suffix + f".{n}")
            dst = self.path.with_suffix(self.path.suffix + f".{n + 1}")
            try:
                src.rename(dst)
            except FileNotFoundError:
                pass
        # Current → .1
        try:
            self.path.rename(self.path.with_suffix(self.path.suffix + ".1"))
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_LOGGER: Optional[EventLogger] = None


def get_event_logger() -> EventLogger:
    """Process-wide singleton — keeps the parent-ready cache hot."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = EventLogger()
    return _LOGGER


def reset_event_logger() -> None:
    """Test seam — drop the singleton so the next ``get_event_logger`` call
    picks up a fresh ``SKEIN_EVENTS_PATH``."""
    global _LOGGER
    _LOGGER = None


def log_event(event: str, scope: Optional[str] = None, **fields: Any) -> None:
    """Convenience wrapper around the singleton."""
    get_event_logger().log(event, scope=scope, **fields)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    # Local-naive RFC3339; matches storage._now_iso() format
    return time.strftime("%Y-%m-%dT%H:%M:%S")
