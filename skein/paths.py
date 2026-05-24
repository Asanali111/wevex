"""Cross-platform location of Skein's per-user state directory.

History
-------
For iterations 1-26 Skein hardcoded ``Path.home() / ".config" / "skein"`` in
~10 modules. That directory holds:

  * ``skein.db``      – the SQLite store
  * ``config.json``   – per-user config (port, embedding provider, token)
  * ``.env``          – optional API-key file (sourced at config load)
  * ``connections.json`` – which LLM clients are connected
  * ``projects.json`` – registry of active projects
  * ``daemon.pid``    – PID file for the nohup backend
  * ``logs/``         – daemon + watcher stdout/stderr
  * ``watchers/``     – per-project watcher PID files

This worked on macOS and Linux because both honor the XDG-ish ``~/.config``
convention. On Windows there is no ``~/.config`` — the per-user state root
is ``%APPDATA%`` (Roaming) for things that should follow the user across
machines, or ``%LOCALAPPDATA%`` for machine-local cache. Skein's DB is
small (~40 MB), portable, and the user moving machines would reasonably
expect their fragments to come along, so we use ``%APPDATA%`` (Roaming).

Design choices (deliberate)
---------------------------
1. **macOS/Linux paths are unchanged.** ``~/.config/skein/`` stays exactly
   where it was. No migration, no surprise relocations. The fleet of
   existing developer installs (n=1: me) continues to work bit-identically.
2. **Windows uses ``%APPDATA%\\skein\\``** with a ``Path.home() / "AppData"
   / "Roaming" / "skein"`` fallback when the env var is not set (some CI
   runners and minimal containers don't set ``APPDATA``).
3. **No ``platformdirs`` dependency.** ``platformdirs.user_data_dir`` would
   put us under ``~/Library/Application Support/skein/`` on macOS — that
   silently relocates the existing live DB. Branching by OS keeps the
   blast radius to Windows only.
4. **Functions, not constants.** Returning ``Path`` from a function lets
   tests monkeypatch ``HOME`` or ``APPDATA`` and re-call. The handful of
   module-level constants in ``daemon.py`` and ``watcher_manager.py``
   still resolve at import time (existing behavior); they read from
   ``skein_home()`` once, which is fine for production but means tests
   that want to override the location must do so before importing those
   modules — same constraint as before.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_windows() -> bool:
    return sys.platform.startswith("win") or os.name == "nt"


def skein_home() -> Path:
    """Return the per-user Skein state directory.

    * Windows:        ``%APPDATA%\\skein\\``   (fallback: ``~/AppData/Roaming/skein/``)
    * macOS / Linux:  ``~/.config/skein/``     (unchanged from pre-iter-27)
    """
    if _is_windows():
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "skein"
        return Path.home() / "AppData" / "Roaming" / "skein"
    return Path.home() / ".config" / "skein"


def default_db_path() -> Path:
    return skein_home() / "skein.db"


def default_config_path() -> Path:
    return skein_home() / "config.json"


def default_env_file() -> Path:
    return skein_home() / ".env"


def daemon_pid_file() -> Path:
    return skein_home() / "daemon.pid"


def daemon_lock_file() -> Path:
    return skein_home() / "daemon.lock"


def daemon_log_dir() -> Path:
    return skein_home() / "logs"


def watcher_pid_dir() -> Path:
    return skein_home() / "watchers"


def watcher_log_dir() -> Path:
    return skein_home() / "logs"


def connections_path() -> Path:
    return skein_home() / "connections.json"


def projects_registry_path() -> Path:
    return skein_home() / "projects.json"


def events_jsonl_path() -> Path:
    return skein_home() / "events.jsonl"


def backend_cache_file() -> Path:
    return skein_home() / "backend"
