"""Pytest gate for the latency scenarios.

Asserts that the ephemeral Skein hits the budgets declared in ``bench/budgets.py``.
A regression that blows past those budgets will fail this test.
"""
from __future__ import annotations

from bench.budgets import evaluate
from bench.scenarios.latency import (
    measure_ingest_throughput,
    measure_recall_latency,
    measure_search_latency,
    measure_seed_throughput,
)


def _assert_budget_ok(scenario_name: str, metrics: dict):
    evals = evaluate(scenario_name, metrics)
    misses = [(m, v) for m, v in evals.items() if not v["ok"]]
    assert not misses, (
        f"{scenario_name} blew budget: "
        + ", ".join(f"{m}={v['observed']:.2f} (budget {v['op']} {v['threshold']})"
                    for m, v in misses)
    )


def test_recall_latency_budget(ephemeral_adapter):
    ephemeral_adapter.remember("Use Redis for session caching",
                               type="decision", scope="project:bench")
    result = measure_recall_latency(ephemeral_adapter, scope="project:bench")
    assert result.status in ("pass", "warn"), result.reason
    _assert_budget_ok("recall_latency", result.metrics)


def test_search_latency_budget(ephemeral_adapter):
    ephemeral_adapter.ingest_text(
        {"a.py": "def foo():\n    return 1\n"}, scope="project:bench",
    )
    result = measure_search_latency(ephemeral_adapter, scope="project:bench")
    assert result.status in ("pass", "warn"), result.reason
    _assert_budget_ok("search_latency", result.metrics)


def test_ingest_throughput_budget(ephemeral_adapter):
    result = measure_ingest_throughput(ephemeral_adapter, scope="project:bench")
    assert result.status in ("pass", "warn"), result.reason
    _assert_budget_ok("ingest_throughput", result.metrics)


def test_seed_throughput_budget(ephemeral_adapter):
    result = measure_seed_throughput(ephemeral_adapter, scope="project:bench")
    assert result.status in ("pass", "warn"), result.reason
    _assert_budget_ok("fragment_write_throughput", result.metrics)
