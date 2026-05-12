"""Retrieval quality — labeled queries against a seeded fragment corpus.

Measures hit@1, hit@5, MRR. Seeds the 25-fragment corpus, runs the 12
labeled queries, and matches returned fragment content against the labels.

Implementation note: ``adapter.remember`` returns a tool-assigned id we cannot
predict. We map our label IDs (``f01``…) to the adapter IDs by remembering
the mapping at seed time, then look up by that map.
"""
from __future__ import annotations

from typing import Dict, List

from ..adapter import MutableAdapter
from ..corpus import fragments, labeled_queries
from ..scenarios import ScenarioResult


def _seed_fragments(adapter: MutableAdapter, *, scope: str) -> Dict[str, str]:
    """Return ``{label_id: adapter_id}`` mapping."""
    mapping: Dict[str, str] = {}
    for f in fragments():
        adapter_id = adapter.remember(
            f["content"], type=f["type"], scope=scope, tags=f.get("tags", []),
        )
        mapping[f["id"]] = adapter_id
    return mapping


def _hit_at_k(
    adapter: MutableAdapter,
    *,
    scope: str,
    label_to_adapter: Dict[str, str],
    k_values: List[int] = (1, 3, 5),
) -> ScenarioResult:
    queries = labeled_queries()
    hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    per_query: List[dict] = []

    for q in queries:
        results = adapter.recall(q["query"], scope=scope, limit=max(k_values))
        ids_returned = [r.id for r in results]
        expected_top = label_to_adapter.get(q["expected_top"])
        expected_set = {label_to_adapter[lid] for lid in q.get("expected_in_top5", [])
                        if lid in label_to_adapter}

        # hit@k = expected_top appears in top-k
        for k in k_values:
            if expected_top in ids_returned[:k]:
                hits[k] += 1

        # Reciprocal rank for MRR — first rank of any expected_in_top5 id
        rr = 0.0
        for rank, fid in enumerate(ids_returned, start=1):
            if fid in expected_set:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        per_query.append({
            "query_id": q["id"],
            "expected_top_rank": (
                ids_returned.index(expected_top) + 1
                if expected_top and expected_top in ids_returned
                else None
            ),
            "reciprocal_rank": rr,
        })

    n = len(queries) or 1
    metrics = {f"hit_at_{k}": hits[k] / n for k in k_values}
    metrics["mrr"] = mrr_sum / n
    metrics["n_queries"] = float(n)

    # Per-difficulty breakdown — distinguishes 'engine broken' from
    # 'embedding provider is bm25-tier'. Easy queries share tokens with
    # their target; hard queries require paraphrase robustness.
    by_diff: Dict[str, List[bool]] = {}
    for q, row in zip(queries, per_query):
        d = q.get("difficulty", "unknown")
        # success at "hit_at_5" granularity
        ok = row["expected_top_rank"] is not None and row["expected_top_rank"] <= 5
        by_diff.setdefault(d, []).append(ok)
    for d, oks in by_diff.items():
        metrics[f"hit_at_5_{d}"] = sum(oks) / len(oks)

    # Floor: at minimum *easy* queries (keyword overlap with target) must
    # work — otherwise the retrieval engine itself is broken. Hard-query
    # failure is informational (= "configure a semantic embedding provider").
    status = "pass"
    reason = ""
    easy_score = metrics.get("hit_at_5_easy", 1.0)
    if easy_score < 0.8:
        status, reason = "fail", f"hit@5 on easy queries {easy_score:.2f} below 0.8"
    elif metrics["hit_at_5"] < 0.5:
        status, reason = "warn", f"overall hit@5 {metrics['hit_at_5']:.2f} below 0.5 (configure semantic embeddings)"

    return ScenarioResult(
        name="retrieval_quality",
        category="quality",
        status=status,
        metrics=metrics,
        reason=reason,
        notes={"per_query": per_query},
    )


def measure_retrieval_quality(
    adapter: MutableAdapter, *, scope: str,
) -> ScenarioResult:
    if not isinstance(adapter, MutableAdapter):
        return ScenarioResult(
            name="retrieval_quality", category="quality", status="skipped",
            reason="quality scenario needs a mutable adapter to seed fragments",
        )
    mapping = _seed_fragments(adapter, scope=scope)
    return _hit_at_k(adapter, scope=scope, label_to_adapter=mapping)
