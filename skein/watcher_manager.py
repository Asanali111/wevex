"""Spawn / track / kill per-project watcher subprocesses.

The watcher must live in the *user's session* (not under launchd) so that
on macOS it has full TCC access to read source files in ~/Documents,
~/Desktop, iCloud, etc.  This module spawns ``skein watch`` as a detached
background subprocess and tracks its PID at:

    ~/.config/skein/watchers/<sanitised-source-root>.pid

The watcher is fire-and-forget from the parent's perspective; it survives
the shell that spawned it (``start_new_session=True``), but dies on logout.
``skein up`` re-spawns it on next invocation.

This split — daemon under launchd, watchers in user session — is the
design Phase 3.5 of the project plan landed on after we discovered launchd
processes can't read files inside macOS TCC-protected dirs.
"""
from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .projects import ProjectEntry

logger = logging.getLogger("skein.watcher_manager")

WATCHER_PID_DIR = Path.home() / ".config" / "skein" / "watchers"
WATCHER_LOG_DIR = Path.home() / ".config" / "skein" / "logs"


def _slug(text: str) -> str:
    """Filesystem-safe single-segment slug."""
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", text).strip("-") or "default"


def pid_file_for(entry: ProjectEntry) -> Path:
    return WATCHER_PID_DIR / f"{_slug(entry.scope)}.pid"


def log_file_for(entry: ProjectEntry) -> Path:
    return WATCHER_LOG_DIR / f"watcher-{_slug(entry.scope)}.log"


def _read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # process exists but we don't own it (rare here)


def is_running(entry: ProjectEntry) -> bool:
    pid_file = pid_file_for(entry)
    if not pid_file.exists():
        return False
    pid = _read_pid(pid_file)
    if pid is None:
        return False
    if not _alive(pid):
        try:
            pid_file.unlink()
        except OSError:
            pass
        return False
    return True


def spawn(entry: ProjectEntry, *, skein_bin: str | None = None) -> int | None:
    """Spawn a detached ``skein watch`` for this project.

    Returns the new PID, or None if a watcher is already running.
    """
    if is_running(entry):
        return None

    WATCHER_PID_DIR.mkdir(parents=True, exist_ok=True)
    WATCHER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    skein_bin = skein_bin or sys.argv[0]
    if not Path(skein_bin).is_file():
        # ``sys.argv[0]`` may be just "skein" if invoked from PATH
        import shutil
        skein_bin = shutil.which("skein") or skein_bin

    log_file = log_file_for(entry)
    cmd = [
        skein_bin, "watch",
        entry.root,
        "--scope", entry.scope,
        "--source-root", entry.source_root,
    ]
    # The child inherits a dup of the log FD; the parent's handle must close
    # after Popen or each spawn leaks one FD into the daemon process forever.
    with open(log_file, "ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    pid_file_for(entry).write_text(str(proc.pid))
    return proc.pid


def kill(entry: ProjectEntry) -> bool:
    """Stop the watcher for one project. Returns True if anything was killed."""
    pid_file = pid_file_for(entry)
    pid = _read_pid(pid_file)
    if pid is None:
        try:
            pid_file.unlink()
        except OSError:
            pass
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait briefly for graceful shutdown
        for _ in range(20):
            if not _alive(pid):
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except ProcessLookupError:
        pass
    finally:
        try:
            pid_file.unlink()
        except OSError:
            pass
    return True


def kill_all() -> list[ProjectEntry]:
    """Stop every active watcher. Returns the entries that were running."""
    from .projects import list_projects
    killed = []
    for entry in list_projects():
        if is_running(entry) and kill(entry):
            killed.append(entry)
    return killed
