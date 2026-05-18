"""Cross-platform detached-process helpers.

Phase 2 of the Windows port (iter 27): every place that spawned a detached
``skein serve`` or ``skein watch`` did so with ``subprocess.Popen(...,
start_new_session=True)`` and stopped it with ``os.kill(pid, SIGTERM)``
followed by ``SIGKILL``. Both are POSIX-only:

  * ``start_new_session`` is a no-op (and a silently-ignored kwarg) on
    Windows; the child inherits the parent's console and dies when the
    spawning terminal closes.
  * ``os.kill(pid, signal.SIGTERM)`` raises ``OSError("Invalid argument")``
    on Windows. The same goes for ``signal.SIGKILL`` (no such signal).
    ``os.kill(pid, 0)`` for liveness probing is also rejected.

This module centralizes the three primitives those call sites need:

  * :func:`spawn_detached`  – fire-and-forget background spawn
  * :func:`terminate_pid`   – graceful shutdown with hard-kill fallback
  * :func:`pid_alive`       – liveness probe

POSIX implementation is the existing nohup recipe (start_new_session +
SIGTERM/SIGKILL). Windows implementation uses ``CREATE_NEW_PROCESS_GROUP |
DETACHED_PROCESS | CREATE_NO_WINDOW`` for spawn, ``CTRL_BREAK_EVENT``
for graceful shutdown, and ``TerminateProcess`` via ctypes for hard kill.
Liveness on Windows opens the process handle with
``PROCESS_QUERY_LIMITED_INFORMATION`` and asks for the exit code — if it's
not ``STILL_ACTIVE``, the process is gone.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Optional, Sequence, Union

_FileLike = Optional[Union[int, IO]]


def _is_windows() -> bool:
    return sys.platform.startswith("win") or os.name == "nt"


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------

def spawn_detached(
    argv: Sequence[str],
    *,
    stdout: _FileLike = None,
    stderr: _FileLike = None,
    env: Optional[dict] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> int:
    """Spawn ``argv`` as a detached background process. Return its PID.

    The child:
      * survives the parent shell closing (no controlling terminal on POSIX,
        no parent console on Windows)
      * has stdin redirected to ``/dev/null`` / ``NUL``
      * sends stdout/stderr wherever the caller asked (default: discard)

    On Windows the child is spawned with ``CREATE_NEW_PROCESS_GROUP`` so that
    :func:`terminate_pid` can send ``CTRL_BREAK_EVENT`` to its group.
    """
    argv_list = [str(a) for a in argv]
    stdout = stdout if stdout is not None else subprocess.DEVNULL
    stderr = stderr if stderr is not None else subprocess.DEVNULL

    if _is_windows():
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
        proc = subprocess.Popen(
            argv_list,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            close_fds=False,  # Windows + redirected handles want False
            env=env,
            cwd=str(cwd) if cwd is not None else None,
        )
    else:
        proc = subprocess.Popen(
            argv_list,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
        )
    return proc.pid


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """Return True iff a process with ``pid`` is currently running."""
    if pid <= 0:
        return False
    if _is_windows():
        return _pid_alive_windows(pid)
    return _pid_alive_posix(pid)


def _pid_alive_posix(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it (rare for our use case — both
        # daemon and watcher are spawned by the same user).
        return True


def _pid_alive_windows(pid: int) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
        )
        if not handle:
            return False
        try:
            code = wintypes.DWORD(0)
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            if not ok:
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Terminate
# ---------------------------------------------------------------------------

def terminate_pid(pid: int, *, timeout: float = 2.0) -> bool:
    """Best-effort terminate. Returns True if the process is gone afterwards.

    Sequence:
      * POSIX: ``SIGTERM`` → wait ``timeout`` seconds → ``SIGKILL``.
      * Windows: ``CTRL_BREAK_EVENT`` → wait ``timeout`` seconds →
        ``TerminateProcess``.

    Treats "process already gone" as success. Never raises on the
    "missing/permission denied" common paths.
    """
    if pid <= 0:
        return True
    if _is_windows():
        return _terminate_windows(pid, timeout=timeout)
    return _terminate_posix(pid, timeout=timeout)


def _terminate_posix(pid: int, *, timeout: float) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive_posix(pid):
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return not _pid_alive_posix(pid)


def _terminate_windows(pid: int, *, timeout: float) -> bool:
    # 1) Try a graceful CTRL_BREAK_EVENT. Only works if the child was spawned
    #    with CREATE_NEW_PROCESS_GROUP, which `spawn_detached` guarantees.
    sigbreak = getattr(signal, "CTRL_BREAK_EVENT", None)
    if sigbreak is not None:
        try:
            os.kill(pid, sigbreak)
        except (ProcessLookupError, OSError):
            # CTRL_BREAK only works on same-console process groups; if it
            # fails we fall through to TerminateProcess.
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive_windows(pid):
            return True
        time.sleep(0.1)
    # 2) Hard kill.
    try:
        import ctypes

        PROCESS_TERMINATE = 0x0001
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if handle:
            try:
                kernel32.TerminateProcess(handle, 1)
            finally:
                kernel32.CloseHandle(handle)
    except OSError:
        return False
    return not _pid_alive_windows(pid)
