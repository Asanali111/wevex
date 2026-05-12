"""Pytest gate for retrieval quality.

The minimum bar is hit@5 = 100% on the *easy* difficulty bucket. If a
regression breaks BM25 or the retrieval pipeline, this fails. We do **not**
assert anything about hard queries — those depend on the configured embedding
provider, which is bm25-only by default in tests.
"""
from __future__ import annotations

from bench.scenarios.retrieval_quality import measure_retrieval_quality


def test_easy_queries_all_hit_top5(ephemeral_adapter):
    result = measure_retrieval_quality(ephemeral_adapter, scope="project:bench")
    assert result.status != "error", result.reason
    easy = result.metrics.get("hit_at_5_easy", 0.0)
    assert easy >= 1.0, (
        f"easy-difficulty hit@5 dropped to {easy:.2f} — retrieval engine regression. "
        f"Per-query: {result.notes.get('per_query')}"
    )


def test_overall_hit_at_5_minimum(ephemeral_adapter):
    """Sanity floor across all difficulties — should never fall below 0.4
    even on bm25-only (easy bucket alone clears it)."""
    result = measure_retrieval_quality(ephemeral_adapter, scope="project:bench")
    assert result.metrics["hit_at_5"] >= 0.4, (
        f"overall hit@5 below 0.4 ({result.metrics['hit_at_5']:.2f})"
    )
