"""Run all scenarios against an adapter, return a structured report."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .adapter import MutableAdapter, ReadOnlyAdapter
from .budgets import evaluate
from .scenarios import ScenarioResult
from .scenarios.auto_capture import measure_auto_capture_quality
from .scenarios.correctness import all_correctness_scenarios
from .scenarios.latency import all_latency_scenarios
from .scenarios.retrieval_quality import measure_retrieval_quality


@dataclass
class BenchmarkReport:
    adapter_name: str
    health: Dict[str, Any]
    scenarios: List[ScenarioResult] = field(default_factory=list)
    budget_evaluations: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    @property
    def overall_status(self) -> str:
        # Order: error > fail > warn > pass; skipped doesn't downgrade.
        order = {"pass": 0, "skipped": 0, "warn": 1, "fail": 2, "error": 3}
        worst = max((order[s.status] for s in self.scenarios), default=0)
        for s in self.scenarios:
            if any(not v["ok"] for v in self.budget_evaluations.get(s.name, {}).values()):
                worst = max(worst, 2)  # budget miss = fail
        return {0: "pass", 1: "warn", 2: "fail", 3: "error"}[worst]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "health": self.health,
            "overall_status": self.overall_status,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "budget_evaluations": self.budget_evaluations,
        }


def _run_safely(name: str, category: str, fn) -> ScenarioResult:
    try:
        return fn()
    except Exception as e:
        return ScenarioResult(
            name=name, category=category, status="error", reason=f"{type(e).__name__}: {e}",
        )


def run(
    adapter: ReadOnlyAdapter,
    *,
    scope: str = "project:bench",
    include_quality: bool = True,
    include_correctness: bool = True,
    include_auto_capture: bool = True,
) -> BenchmarkReport:
    """Run all eligible scenarios. Returns a structured report.

    Read-only adapters skip mutable scenarios automatically; the report records
    them as ``skipped`` with a reason.
    """
    health = adapter.health()
    scenarios: List[ScenarioResult] = []

    # If mutable, run a clean reset first so seeded scenarios start fresh.
    if isinstance(adapter, MutableAdapter):
        adapter.reset()
        adapter.ensure_scope(scope)

    # Order matters: ingest_throughput seeds chunks needed by search_latency;
    # seed_throughput seeds fragments needed by recall_latency.
    if isinstance(adapter, MutableAdapter):
        scenarios.extend(_seed_then_measure_latency(adapter, scope=scope))
    else:
        scenarios.extend(all_latency_scenarios(adapter, scope=scope))

    if include_quality and isinstance(adapter, MutableAdapter):
        # Reset before quality so seed state is deterministic.
        adapter.reset()
        adapter.ensure_scope(scope)
        scenarios.append(_run_safely(
            "retrieval_quality", "quality",
            lambda: measure_retrieval_quality(adapter, scope=scope),
        ))

    if include_auto_capture and isinstance(adapter, MutableAdapter):
        scenarios.append(_run_safely(
            "auto_capture_quality", "quality",
            lambda: measure_auto_capture_quality(adapter),
        ))

    if include_correctness and isinstance(adapter, MutableAdapter):
        adapter.reset()
        for r in all_correctness_scenarios(adapter):
            scenarios.append(r)

    # Budget evaluation
    budget = {s.name: evaluate(s.name, s.metrics) for s in scenarios}
    # Any budget miss downgrades the scenario status to fail if currently pass.
    for s in scenarios:
        misses = [m for m, v in budget.get(s.name, {}).items() if not v["ok"]]
        if misses and s.status == "pass":
            s.status = "fail"
            s.reason = f"budget miss: {', '.join(misses)}"

    return BenchmarkReport(
        adapter_name=adapter.name,
        health=health.__dict__,
        scenarios=scenarios,
        budget_evaluations=budget,
    )


def _seed_then_measure_latency(
    adapter: MutableAdapter, *, scope: str,
) -> List[ScenarioResult]:
    """Ingest + seed first (their throughput numbers are also useful), then
    measure recall/search latency against the seeded state."""
    from .scenarios.latency import (
        measure_ingest_throughput, measure_recall_latency,
        measure_search_latency, measure_seed_throughput,
    )

    out: List[ScenarioResult] = []
    out.append(_run_safely("ingest_throughput", "latency",
                           lambda: measure_ingest_throughput(adapter, scope=scope)))
    out.append(_run_safely("fragment_write_throughput", "latency",
                           lambda: measure_seed_throughput(adapter, scope=scope)))
    out.append(_run_safely("recall_latency", "latency",
                           lambda: measure_recall_latency(adapter, scope=scope)))
    out.append(_run_safely("search_latency", "latency",
                           lambda: measure_search_latency(adapter, scope=scope)))
    # Iter 31: token-budget scenario — regression guard against future
    # changes that re-bloat the recall payload.
    from .scenarios.token_budget import measure_recall_token_budget
    out.append(_run_safely("recall_token_budget", "efficiency",
                           lambda: measure_recall_token_budget(adapter, scope=scope)))
    return out
