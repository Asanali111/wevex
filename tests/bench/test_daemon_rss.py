"""Smoke tests for the daemon_rss bench scenario.

These exercise the shape of the result on an in-process ephemeral adapter.
The actual 50 MB RSS-drop floor only matters against a live daemon with a
real FastembedProvider — see ``bench/budgets.py::daemon_rss`` and the
scenario's module docstring for the live-mode story.

The default ephemeral adapter uses the "hash" embedding provider, which
has no ``idle_check_and_unload`` method. That hits the no-unload-hook
branch and returns ``status="warn"`` with informational metrics, which
is exactly what we want the budget to NOT flip into a fail. A second
test confirms that branch.
"""
from __future__ import annotations

import platform
import sys
from unittest.mock import patch

import pytest

from bench.scenarios.daemon_rss import measure_daemon_rss


# The scenario sleeps for ~15 s by default. We monkey-patch the module
# constant down to 1 s for unit tests — the shape of the result is what
# we're verifying, not the actual idle-unload mechanism (which is a
# live-mode-only concern by design).
@pytest.fixture(autouse=True)
def _shorten_sleep(monkeypatch):
    monkeypatch.setattr("bench.scenarios.daemon_rss._SLEEP_SECONDS", 1)
    monkeypatch.setattr("bench.scenarios.daemon_rss._IDLE_SECONDS", 1)


@pytest.mark.skipif(
    platform.system().lower().startswith("win"),
    reason="daemon_rss scenario skips on Windows; no ps available",
)
def test_daemon_rss_runs_to_completion(ephemeral_adapter):
    """Scenario must return a ScenarioResult with the expected fields,
    not raise. The ephemeral adapter's default provider has no unload
    hook, so we expect status='warn' and zero drop — but the result
    shape must still be correct."""
    result = measure_daemon_rss(ephemeral_adapter, scope="project:bench")
    assert result.name == "daemon_rss"
    assert result.category == "efficiency"
    assert result.status in ("pass", "warn", "skipped")


@pytest.mark.skipif(
    platform.system().lower().startswith("win"),
    reason="daemon_rss scenario skips on Windows",
)
def test_daemon_rss_metrics_have_plausible_shape(ephemeral_adapter):
    """When the scenario doesn't skip, it must report ``pre_idle_rss_mb``
    + ``idle_seconds`` with non-negative floats, plus either the budgeted
    pair (``post_idle_rss_mb`` + ``idle_rss_drop_mb``) when an unload ran,
    or the unbudgeted observation pair otherwise."""
    result = measure_daemon_rss(ephemeral_adapter, scope="project:bench")
    if result.status == "skipped":
        # Fine — environment couldn't read RSS. Don't fail; the live
        # mode is where the real floor lives anyway.
        return
    m = result.metrics
    # Always present:
    for key in ("pre_idle_rss_mb", "idle_seconds"):
        assert key in m, f"missing metric: {key}"
        assert isinstance(m[key], float), f"{key} must be float"
        assert m[key] >= 0.0, f"{key} must be non-negative, got {m[key]}"
    # One of two metric families must be present depending on whether
    # an unload check actually ran.
    has_budgeted = "post_idle_rss_mb" in m and "idle_rss_drop_mb" in m
    has_unbudgeted = (
        "post_idle_rss_mb_unbudgeted" in m and "rss_observed_drop_mb" in m
    )
    assert has_budgeted or has_unbudgeted, (
        f"metrics missing post-idle pair: keys={list(m.keys())}"
    )
    # Recall-process RSS should be at least a few MB on any real
    # Python interpreter — sanity floor that catches misreads.
    assert m["pre_idle_rss_mb"] > 1.0, (
        f"pre_idle_rss_mb suspiciously low: {m['pre_idle_rss_mb']}"
    )


@pytest.mark.skipif(
    platform.system().lower().startswith("win"),
    reason="daemon_rss scenario skips on Windows",
)
def test_daemon_rss_no_unload_hook_returns_warn(ephemeral_adapter):
    """The ephemeral adapter uses the hash embedding provider which has
    no idle_check_and_unload method. The scenario MUST return
    status='warn' (not 'pass') in this case so the budget evaluator
    doesn't flip a zero-drop result into 'fail'."""
    # The ephemeral adapter's `_provider` is HashEmbeddingProvider by
    # default — no unload hook. Confirm by introspection.
    provider = getattr(ephemeral_adapter, "_provider", None)
    assert provider is not None, (
        "ephemeral adapter must expose its embedding provider"
    )
    assert not hasattr(provider, "idle_check_and_unload"), (
        "test assumption: hash provider has no idle_check_and_unload"
    )

    result = measure_daemon_rss(ephemeral_adapter, scope="project:bench")
    # Could be 'skipped' on a stripped-down CI box that can't read RSS;
    # the meaningful contract is "not 'pass' and not 'fail'/'error'".
    assert result.status in ("warn", "skipped"), (
        f"expected warn/skipped without unload hook, got {result.status} "
        f"reason={result.reason}"
    )
    if result.status == "warn":
        assert "idle_check_and_unload" in result.reason or "no" in result.reason.lower()


def test_daemon_rss_windows_skips(monkeypatch, ephemeral_adapter):
    """Force the platform check to think we're on Windows and confirm
    the scenario skips cleanly with the expected reason."""
    monkeypatch.setattr(
        "bench.scenarios.daemon_rss.platform.system",
        lambda: "Windows",
    )
    result = measure_daemon_rss(ephemeral_adapter, scope="project:bench")
    assert result.status == "skipped"
    assert "Windows" in result.reason
