"""Default budgets for latency scenarios.

Format: ``BUDGETS[scenario_name][metric] = (op, threshold)`` where ``op``
is one of ``"<="`` / ``">="`` / ``"<"`` / ``">"``. Quality and correctness
scenarios encode their own pass/warn/fail logic inside the scenario module
because the thresholds are tied to interpretation.

Numbers are intentionally generous defaults so that any working local-only
context bus on a developer laptop passes. Tighten in your own CI config
if you want stricter gating.
"""
from __future__ import annotations

from typing import Dict, Tuple

Op = str  # one of "<=", ">=", "<", ">"


BUDGETS: Dict[str, Dict[str, Tuple[Op, float]]] = {
    "recall_latency": {
        "warm_p50_ms": ("<=", 100.0),
        "warm_p95_ms": ("<=", 250.0),
        "cold_p95_ms": ("<=", 2500.0),
    },
    "search_latency": {
        "warm_p50_ms": ("<=", 200.0),
        "warm_p95_ms": ("<=", 1000.0),
        "cold_p95_ms": ("<=", 5000.0),
    },
    "ingest_throughput": {
        "chunks_per_sec": (">=", 5.0),
    },
    "fragment_write_throughput": {
        "write_p95_ms": ("<=", 100.0),
        "fragments_per_sec": (">=", 20.0),
    },
    # Iter 31 efficiency pass: snippet rendering at the MCP layer caps
    # each rendered result around 320 chars (≈80 tokens). This budget
    # measures the underlying fragment content lengths returned by the
    # adapter — a smoke that catches regressions where fragments balloon
    # past the soft 800-char write cap.
    "recall_token_budget": {
        "avg_chars_per_result": ("<=", 1000.0),
        "max_chars_per_result": ("<=", 2000.0),
        "total_chars":          ("<=", 40000.0),
    },
}


def evaluate(scenario: str, metrics: Dict[str, float]) -> Dict[str, dict]:
    """Return ``{metric: {observed, op, threshold, ok}}`` for each budgeted metric."""
    out: Dict[str, dict] = {}
    rules = BUDGETS.get(scenario, {})
    for metric, (op, threshold) in rules.items():
        if metric not in metrics:
            continue
        observed = metrics[metric]
        ok = _compare(observed, op, threshold)
        out[metric] = {
            "observed": observed, "op": op, "threshold": threshold, "ok": ok,
        }
    return out


def _compare(value: float, op: str, threshold: float) -> bool:
    return {
        "<=": value <= threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        ">": value > threshold,
    }[op]
