"""Latency scenarios — pure perf, no labels required.

Measures cold/warm p50/p95 for recall and code search, plus ingest throughput
on the mutable adapter. The live (read-only) adapter skips ingest.
"""
from __future__ import annotations

import time
from typing import Callable

from ..adapter import MutableAdapter, ReadOnlyAdapter
from ..corpus import code_files, fragments
from ..scenarios import ScenarioResult

_RECALL_QUERIES = [
    "database choice", "session caching", "rate limit",
    "authentication tokens", "deployment pipeline", "monitoring",
    "preferences async", "GDPR export", "secret rotation procedure",
]

_SEARCH_QUERIES = [
    "redis session", "authenticate user password", "stripe charge",
    "rate limit per user", "upload file S3", "JWT lifetime",
]


def _time(fn: Callable[[], object]) -> float:
    """Return wall-clock ms for one invocation."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def _pcts(samples: list[float]) -> dict:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    samples = sorted(samples)
    n = len(samples)
    return {
        "p50": samples[n // 2],
        "p95": samples[min(n - 1, int(0.95 * n))],
        "max": samples[-1],
        "n": float(n),
    }


def measure_recall_latency(
    adapter: ReadOnlyAdapter,
    *,
    scope: str,
    queries: list[str] = _RECALL_QUERIES,
    warm_repeats: int = 3,
) -> ScenarioResult:
    """One cold pass, then ``warm_repeats`` repeated passes; report both."""
    cold: list[float] = []
    warm: list[float] = []
    for q in queries:
        cold.append(_time(lambda q=q: adapter.recall(q, scope, limit=5)))
    for _ in range(warm_repeats):
        for q in queries:
            warm.append(_time(lambda q=q: adapter.recall(q, scope, limit=5)))
    c, w = _pcts(cold), _pcts(warm)
    return ScenarioResult(
        name="recall_latency",
        category="latency",
        metrics={
            "cold_p50_ms": c["p50"], "cold_p95_ms": c["p95"], "cold_max_ms": c["max"],
            "warm_p50_ms": w["p50"], "warm_p95_ms": w["p95"], "warm_max_ms": w["max"],
            "samples": c["n"] + w["n"],
        },
    )


def measure_search_latency(
    adapter: ReadOnlyAdapter,
    *,
    scope: str,
    queries: list[str] = _SEARCH_QUERIES,
    warm_repeats: int = 3,
) -> ScenarioResult:
    """Same shape as recall latency, but over the code-search path."""
    if not adapter.supports_code_search:
        return ScenarioResult(
            name="search_latency", category="latency", status="skipped",
            reason="adapter does not declare code-search support",
        )
    cold: list[float] = []
    warm: list[float] = []
    for q in queries:
        cold.append(_time(lambda q=q: adapter.search_code(q, scope, limit=5)))
    for _ in range(warm_repeats):
        for q in queries:
            warm.append(_time(lambda q=q: adapter.search_code(q, scope, limit=5)))
    c, w = _pcts(cold), _pcts(warm)
    return ScenarioResult(
        name="search_latency",
        category="latency",
        metrics={
            "cold_p50_ms": c["p50"], "cold_p95_ms": c["p95"], "cold_max_ms": c["max"],
            "warm_p50_ms": w["p50"], "warm_p95_ms": w["p95"], "warm_max_ms": w["max"],
            "samples": c["n"] + w["n"],
        },
    )


def measure_ingest_throughput(
    adapter: MutableAdapter,
    *,
    scope: str,
) -> ScenarioResult:
    """Index the synthetic code corpus and report files/sec + chunks/sec."""
    if not isinstance(adapter, MutableAdapter):
        return ScenarioResult(
            name="ingest_throughput", category="latency", status="skipped",
            reason="ingest requires a mutable adapter",
        )
    files = code_files()
    t0 = time.perf_counter()
    n_chunks = adapter.ingest_text(files, scope=scope, source_root="bench_ingest")
    elapsed = time.perf_counter() - t0
    if elapsed <= 0:
        elapsed = 1e-6
    return ScenarioResult(
        name="ingest_throughput",
        category="latency",
        metrics={
            "files": float(len(files)),
            "chunks": float(n_chunks),
            "elapsed_ms": elapsed * 1000.0,
            "files_per_sec": len(files) / elapsed,
            "chunks_per_sec": n_chunks / elapsed,
        },
    )


def measure_seed_throughput(
    adapter: MutableAdapter,
    *,
    scope: str,
) -> ScenarioResult:
    """Insert the 25-fragment corpus and report fragments/sec."""
    frags = fragments()
    t0 = time.perf_counter()
    durations: list[float] = []
    for f in frags:
        ts = time.perf_counter()
        adapter.remember(
            f["content"], type=f["type"], scope=scope, tags=f.get("tags", []),
        )
        durations.append((time.perf_counter() - ts) * 1000.0)
    elapsed = time.perf_counter() - t0
    if elapsed <= 0:
        elapsed = 1e-6
    p = _pcts(durations)
    return ScenarioResult(
        name="fragment_write_throughput",
        category="latency",
        metrics={
            "n": float(len(frags)),
            "elapsed_ms": elapsed * 1000.0,
            "fragments_per_sec": len(frags) / elapsed,
            "write_p50_ms": p["p50"],
            "write_p95_ms": p["p95"],
            "write_max_ms": p["max"],
        },
    )


def all_latency_scenarios(
    adapter: ReadOnlyAdapter, *, scope: str,
) -> list[ScenarioResult]:
    out = [
        measure_recall_latency(adapter, scope=scope),
        measure_search_latency(adapter, scope=scope),
    ]
    if isinstance(adapter, MutableAdapter):
        out.append(measure_ingest_throughput(adapter, scope=scope))
        out.append(measure_seed_throughput(adapter, scope=scope))
    return out
