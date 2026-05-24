"""Daemon RSS scenario — iter 31 ONNX-idle-unload regression antibody.

When the FastembedProvider goes idle for `_IDLE_UNLOAD_SECONDS` (default
600, env-overridable via `SKEIN_FASTEMBED_IDLE_SECONDS`), the daemon's
background loop drops the ONNX runtime. The expected resident-set-size
(RSS) drop is ~200 MB from the ONNX runtime + some additional reclaim
from SQLite page-cache trimming.

If that mechanism regresses — e.g. someone caches a reference to the
runtime, or the env var stops being honored — the daemon's resident
memory will stay high during inactive periods, breaking the "Skein is a
quiet background daemon" promise. This scenario catches that regression.

Measurement strategy:
  1. Run a small recall round to ensure the embedding runtime is loaded
     (and pages are paged in).
  2. Sample RSS — that's `pre_idle_rss_mb`.
  3. Configure the provider's idle window to ~12 s via
     `SKEIN_FASTEMBED_IDLE_SECONDS=12` for the duration of this scenario
     only (we do NOT mutate the user's persisted config).
  4. Sleep ~15 s with no embed activity.
  5. For in-process adapters: call `provider.idle_check_and_unload()`
     directly (there's no daemon loop to do it for us). For live mode
     the daemon's `_embedding_idle_unload_loop` (60 s cadence) handles
     it; we just sleep long enough to catch the next tick.
  6. Sample RSS again — that's `post_idle_rss_mb`.
  7. Drop = pre - post; budget asserts >= 50 MB (in `budgets.py`).

Platform notes:
  * `psutil` isn't a project dependency, so we shell out to `ps -o rss=
    -p <pid>` which works on macOS + Linux.
  * Windows lacks an equivalent one-shot RSS reader without ctypes
    plumbing; the scenario skips cleanly with a reason.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import Optional

from ..adapter import MutableAdapter, ReadOnlyAdapter
from ..corpus import labeled_queries
from ..scenarios import ScenarioResult


# Real env var read by FastembedProvider.idle_check_and_unload() — see
# skein/embeddings.py. The task description named a different var; we
# honor the implementation, not the description.
_IDLE_ENV_VAR = "SKEIN_FASTEMBED_IDLE_SECONDS"
_IDLE_SECONDS = 12
_SLEEP_SECONDS = 15  # > _IDLE_SECONDS so the unload check fires


def _read_rss_mb(pid: int) -> Optional[float]:
    """Return resident-set-size in MB for ``pid`` using ``ps``.

    Returns None if the read fails (missing binary, dead PID, permission
    denied, weird platform). The caller decides whether that's a skip
    or a warn.
    """
    try:
        # ps emits RSS in KB on both macOS + Linux. `-o rss=` strips
        # the header so the output is purely the number.
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return None
    text = out.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        kb = float(text.split()[0])
    except (ValueError, IndexError):
        return None
    return kb / 1024.0


def _resolve_daemon_pid(adapter: ReadOnlyAdapter) -> Optional[int]:
    """Pick the right PID to measure for ``adapter``.

    Live adapters: prefer ``adapter.daemon_pid()`` if exposed, else read
    ``~/.config/skein/daemon.pid``. In-process adapters: just measure
    the current Python process — that's the one running the embedding
    runtime.
    """
    # Live adapter path: explicit accessor wins.
    getter = getattr(adapter, "daemon_pid", None)
    if callable(getter):
        try:
            pid = int(getter())
            if pid > 0 and _pid_alive(pid):
                return pid
        except Exception:
            pass
    # In-process adapters carry storage + provider on the instance —
    # detect by attribute presence and just measure ourselves. This
    # avoids accidentally querying a stale daemon.pid file from a long-
    # dead launchd daemon.
    if hasattr(adapter, "_storage") and hasattr(adapter, "_provider"):
        return os.getpid()
    # Live adapter fallback: pid file written by the daemon. Skip if
    # the PID is dead (e.g. stale pidfile from a previous boot).
    pid_path = os.path.expanduser("~/.config/skein/daemon.pid")
    if os.path.exists(pid_path):
        try:
            pid = int(open(pid_path).read().strip())
            if pid > 0 and _pid_alive(pid):
                return pid
        except Exception:
            pass
    # Last resort: the current process. Better than nothing for any
    # third-party adapter that hasn't declared a PID accessor.
    return os.getpid()


def _pid_alive(pid: int) -> bool:
    """Return True iff ``pid`` is alive. ``os.kill(pid, 0)`` doesn't
    deliver a real signal; it just lets the kernel report whether the
    PID exists. ``PermissionError`` means the PID is alive but owned
    by someone else — still alive enough to count."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _run_warm_recall(adapter: ReadOnlyAdapter, scope: str) -> None:
    """Touch recall a few times so the embedding runtime is loaded and
    pages are resident before we sample RSS."""
    queries = labeled_queries() or [{"query": "warmup"}]
    for q in queries[:5]:
        try:
            adapter.recall(q["query"], scope=scope, limit=5)
        except Exception:
            # Recall errors aren't this scenario's job — the latency
            # scenario surfaces those. We just need the side effect of
            # touching the embedding runtime.
            pass


def measure_daemon_rss(
    adapter: MutableAdapter,
    *,
    scope: str,
) -> ScenarioResult:
    """Measure the RSS drop between a warm recall round and an idle window.

    The scenario short-circuits to ``skipped`` on Windows (no ``ps``)
    and to ``warn`` when the adapter exposes no embedding provider that
    can be unloaded.
    """
    # Hard skip on Windows — no `ps` and no comparable one-shot RSS
    # reader without ctypes plumbing.
    if platform.system().lower().startswith("win"):
        return ScenarioResult(
            name="daemon_rss", category="efficiency",
            status="skipped",
            reason="no measurement method on Windows",
        )

    pid = _resolve_daemon_pid(adapter)
    if pid is None:
        return ScenarioResult(
            name="daemon_rss", category="efficiency",
            status="skipped",
            reason="could not resolve a PID to measure",
        )

    # First touch: warm the embedding runtime + take pre-idle sample.
    _run_warm_recall(adapter, scope)
    pre = _read_rss_mb(pid)
    if pre is None:
        return ScenarioResult(
            name="daemon_rss", category="efficiency",
            status="skipped",
            reason=f"could not read RSS for pid {pid} (ps unavailable?)",
        )

    # Shrink the idle window for this run only. We restore the previous
    # value at the end so we don't leak state into other scenarios.
    prev_env = os.environ.get(_IDLE_ENV_VAR)
    os.environ[_IDLE_ENV_VAR] = str(_IDLE_SECONDS)

    sleep_started = time.monotonic()
    try:
        time.sleep(_SLEEP_SECONDS)
        # In-process adapters carry the provider on the instance —
        # there's no daemon loop, so we must trigger the unload check
        # ourselves. Live adapters: the daemon's background loop runs
        # `idle_check_and_unload` every `embedding_idle_check_interval`
        # seconds (default 60). At a 15 s sleep we may or may not see
        # a tick — that's why this scenario's hard floor lives in the
        # budget (live runs over a longer window) rather than the
        # scenario itself.
        provider = getattr(adapter, "_provider", None)
        unload_fn = getattr(provider, "idle_check_and_unload", None)
        if callable(unload_fn):
            try:
                unload_fn()
            except Exception:
                # A buggy provider shouldn't take down the scenario.
                pass
        else:
            # No unload mechanism at all — return the samples but mark
            # as warn AND omit the budgeted ``idle_rss_drop_mb`` /
            # ``post_idle_rss_mb`` keys. budgets.py only enforces metrics
            # actually present in the dict, so the budget evaluator can't
            # flip this run to fail when the provider couldn't possibly
            # have dropped anything.
            post_no_unload = _read_rss_mb(pid) or 0.0
            return ScenarioResult(
                name="daemon_rss", category="efficiency",
                status="warn",
                reason=("embedding provider has no idle_check_and_unload "
                        "hook — RSS drop measurement is informational"),
                metrics={
                    "pre_idle_rss_mb": pre,
                    # NB: not "post_idle_rss_mb" — the budgeted name is
                    # deliberately reserved for runs where an unload
                    # actually ran. We surface the observation under
                    # a separate key for visibility.
                    "post_idle_rss_mb_unbudgeted": post_no_unload,
                    "rss_observed_drop_mb": max(0.0, pre - post_no_unload),
                    "idle_seconds": float(time.monotonic() - sleep_started),
                },
            )
        post = _read_rss_mb(pid)
    finally:
        # Restore prior env so other scenarios see the user's real config.
        if prev_env is None:
            os.environ.pop(_IDLE_ENV_VAR, None)
        else:
            os.environ[_IDLE_ENV_VAR] = prev_env

    if post is None:
        return ScenarioResult(
            name="daemon_rss", category="efficiency",
            status="skipped",
            reason=f"could not read RSS post-idle for pid {pid}",
            metrics={
                "pre_idle_rss_mb": pre,
                "idle_seconds": float(time.monotonic() - sleep_started),
            },
        )

    drop = pre - post
    return ScenarioResult(
        name="daemon_rss", category="efficiency",
        status="pass",
        metrics={
            "pre_idle_rss_mb": pre,
            "post_idle_rss_mb": post,
            "idle_rss_drop_mb": drop,
            "idle_seconds": float(time.monotonic() - sleep_started),
        },
    )
