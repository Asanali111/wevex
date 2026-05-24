"""OS-level single-instance enforcement for ``skein serve``.

Acquires an advisory file lock on ``skein_home() / "daemon.lock"`` before
the daemon starts. The kernel auto-releases the lock when the holding
process dies, so there's no stale-PID cleanup problem.

Without this, a second ``skein serve --port N`` (N != configured port)
happily starts a parallel daemon — that's how the 8766 leftover ran
alongside 8765 for three days in May 2026. The TCP port is the only
implicit "lock" today, and it only protects against same-port collisions.

Unix: ``fcntl.flock(LOCK_EX | LOCK_NB)``.
Windows: ``msvcrt.locking(LK_NBLCK)``.

The returned handle MUST be kept alive for the lifetime of the daemon —
losing the reference closes the file and releases the lock.
"""
from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LockResult:
    """Outcome of an ``acquire`` call.

    ``acquired=True`` → caller owns the lock; ``handle`` is the open file
    descriptor that must be held for the lifetime of the daemon.

    ``acquired=False`` → another daemon holds it; ``existing_pid`` is the
    PID recorded in the lock file (best-effort; may be ``None`` if the
    file was created by an older skein that didn't write a PID).
    """
    acquired: bool
    handle: Optional[object] = None
    existing_pid: Optional[int] = None
    lock_path: Optional[Path] = None


def acquire(lock_path: Path) -> LockResult:
    """Try to acquire an exclusive non-blocking lock on ``lock_path``.

    On success: writes our PID into the file and returns the open handle.
    On contention: reads whatever PID was written by the holder and
    returns ``acquired=False``.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if platform.system() == "Windows":
        return _acquire_windows(lock_path)
    return _acquire_unix(lock_path)


def _acquire_unix(lock_path: Path) -> LockResult:
    import fcntl

    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        existing_pid = _read_pid(fh)
        fh.close()
        return LockResult(acquired=False, existing_pid=existing_pid,
                          lock_path=lock_path)
    _write_pid(fh)
    return LockResult(acquired=True, handle=fh, lock_path=lock_path)


def _acquire_windows(lock_path: Path) -> LockResult:
    import msvcrt  # type: ignore[import]

    fh = open(lock_path, "a+b")
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
    except OSError:
        existing_pid = _read_pid(fh)
        fh.close()
        return LockResult(acquired=False, existing_pid=existing_pid,
                          lock_path=lock_path)
    _write_pid(fh)
    return LockResult(acquired=True, handle=fh, lock_path=lock_path)


def _write_pid(fh) -> None:
    try:
        fh.seek(0)
        fh.truncate()
        if "b" in (getattr(fh, "mode", "") or ""):
            fh.write(f"{os.getpid()}\n".encode("utf-8"))
        else:
            fh.write(f"{os.getpid()}\n")
        fh.flush()
        os.fsync(fh.fileno())
    except OSError:
        # PID-recording failure is informational, not fatal — the lock
        # itself is what matters. A second daemon attempt will still be
        # blocked; it just won't see the holder PID.
        logger.debug("failed to write PID into lock file", exc_info=True)


def _read_pid(fh) -> Optional[int]:
    try:
        fh.seek(0)
        raw = fh.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        first = raw.strip().splitlines()[0] if raw.strip() else ""
        return int(first) if first else None
    except (OSError, ValueError, IndexError):
        return None


def release(handle) -> None:
    """Release the lock and close the file. Idempotent."""
    if handle is None:
        return
    try:
        if platform.system() == "Windows":
            import msvcrt  # type: ignore[import]
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


def acquire_or_exit(lock_path: Path, *, stderr=None) -> object:
    """Convenience: acquire the lock or exit cleanly with a clear message.

    Exit code is **0** on contention, not non-zero. Reason: under
    ``launchd``'s ``KeepAlive=true`` an unsuccessful exit triggers an
    immediate respawn loop. Treating "another daemon already has it" as
    a successful no-op lets launchd settle without thrashing.

    Returns the lock handle on success. Callers must keep the reference
    alive for the lifetime of the daemon (the lock dies with the file
    descriptor).
    """
    out = stderr if stderr is not None else sys.stderr
    result = acquire(lock_path)
    if result.acquired:
        return result.handle
    pid_str = f"PID {result.existing_pid}" if result.existing_pid else "unknown PID"
    print(
        f"skein: another daemon is already running ({pid_str}). "
        f"Use `skein down` to stop it first, then `skein up` again. "
        f"Lock file: {result.lock_path}",
        file=out,
    )
    sys.exit(0)
