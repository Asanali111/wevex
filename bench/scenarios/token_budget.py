"""Token-budget scenario — iter 31 regression antibody.

Skein's first-class promise is "more context, fewer tokens". When that
ratio flips — when a single ``recall(limit=10)`` ships kilobytes of text
into the LLM's prompt — speculative recall calls get expensive, agents
stop reaching for the tool, and Skein silently loses its place in the
LLM's toolkit. This scenario catches that regression before it ships.

Measures, against the seeded 25-fragment corpus over the 12 labeled
queries:
  - avg_chars_per_result   — mean across all rendered results
  - total_chars            — sum across all queries × limit
  - max_chars_per_result   — worst-case (catches the "one fragment
                              dumped a 4 000-char wall of text" case)

Budget is enforced via bench/budgets.py.
"""
from __future__ import annotations

from statistics import mean
from typing import List

from ..adapter import MutableAdapter
from ..corpus import labeled_queries
from ..scenarios import ScenarioResult


def measure_recall_token_budget(
    adapter: MutableAdapter, *, scope: str, limit: int = 5,
) -> ScenarioResult:
    """Run every labeled query through ``adapter.recall`` and measure
    the rendered character count of each result. Adapters return
    fragment objects; we measure ``.content`` length because that's
    what the MCP layer renders (with snippet truncation under iter 31).
    """
    queries = labeled_queries()
    if not queries:
        return ScenarioResult(
            name="recall_token_budget", category="efficiency",
            status="skipped",
            reason="no labeled queries in corpus",
        )

    all_lens: List[int] = []
    per_query_total = 0
    for q in queries:
        results = adapter.recall(q["query"], scope=scope, limit=limit)
        for r in results:
            content = getattr(r, "content", "") or ""
            all_lens.append(len(content))
            per_query_total += len(content)

    if not all_lens:
        return ScenarioResult(
            name="recall_token_budget", category="efficiency",
            status="warn",
            reason="recall returned zero results across all labeled queries",
            metrics={
                "avg_chars_per_result": 0.0,
                "total_chars": 0.0,
                "max_chars_per_result": 0.0,
            },
        )

    return ScenarioResult(
        name="recall_token_budget", category="efficiency",
        status="pass",
        metrics={
            "avg_chars_per_result": float(mean(all_lens)),
            "total_chars": float(per_query_total),
            "max_chars_per_result": float(max(all_lens)),
            "samples": float(len(all_lens)),
        },
    )
