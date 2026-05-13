"""Cross-session handoff — the core "context bus" value-prop.

Simulates two LLM tools sharing one Skein store: tool A writes a fragment
in one session, tool B (later) recalls it without the user re-pasting
context. We don't actually fork a subprocess — the "session boundary" is
conceptual. What we measure is whether a fragment written by one
``source_tool`` is retrievable by a query issued from a *different*
``source_tool``.

Why this matters: every Skein pitch leans on "tool A captures, tool B
recalls." A bench that doesn't verify it leaves the headline claim
unmeasured.

The scenario requires the adapter to accept a ``source_tool`` kwarg on
both ``remember`` and ``recall``. If it doesn't, we skip — and the
report names the API addition that would unblock the measurement.
"""
from __future__ import annotations

import time
from typing import Any

from ..adapter import MutableAdapter
from ..corpus import fragments, labeled_queries
from ..scenarios import ScenarioResult

_SCOPE = "project:bench-handoff"
_WRITE_TOOL = "claude_code"
_QUERY_TOOL = "cursor"
_N_FRAGMENTS = 5


def _query_for_fragment(
    fragment: dict[str, Any], by_expected_top: dict[str, dict[str, Any]],
) -> str:
    """Pick a labeled query targeting this fragment, else synthesize one.

    Synthesis: first 5 words of the content + the first tag (if any).
    Cheap, deterministic, and biased toward keyword overlap so the
    measurement isn't dominated by embedding quality — we want to
    measure handoff, not retrieval-engine recall.
    """
    fid = fragment["id"]
    labeled = by_expected_top.get(fid)
    if labeled and labeled.get("query"):
        return labeled["query"]
    words = (fragment.get("content") or "").split()[:5]
    tag = (fragment.get("tags") or [None])[0]
    parts = list(words) + ([tag] if tag else [])
    return " ".join(p for p in parts if p)


def _supports_source_tool(adapter: MutableAdapter, scope: str) -> str | None:
    """Probe whether the adapter accepts ``source_tool`` on remember + recall.

    Returns ``None`` on success, else a human-readable reason string
    naming the missing API. We try the smallest sentinel write and a
    one-shot recall; cleanup of the sentinel is best-effort (the scope
    is ephemeral anyway).
    """
    try:
        adapter.remember(
            "sentinel: cross-session handoff probe",
            type="fact",
            scope=scope,
            tags=["bench-handoff-probe"],
            source_tool=_WRITE_TOOL,
        )
    except TypeError:
        return (
            "MutableAdapter.remember does not accept a `source_tool` kwarg. "
            "Unblock: add `source_tool: Optional[str] = None` to "
            "remember()/recall() on bench/adapter.py, persist it on the "
            "fragment row, and surface it on FragmentResult so the scenario "
            "can verify writes from tool A are retrieved by tool B."
        )
    except NotImplementedError as e:
        return f"adapter declined source-tool probe: {e}"
    try:
        adapter.recall("sentinel", scope=scope, limit=1, source_tool=_QUERY_TOOL)
    except TypeError:
        return (
            "MutableAdapter.recall does not accept a `source_tool` kwarg. "
            "Unblock: add `source_tool: Optional[str] = None` to recall() "
            "on bench/adapter.py so callers can attribute the querying tool."
        )
    except NotImplementedError as e:
        return f"adapter declined source-tool recall: {e}"
    return None


def measure_cross_session_handoff(adapter: MutableAdapter) -> ScenarioResult:
    if not isinstance(adapter, MutableAdapter):
        return ScenarioResult(
            name="cross_session_handoff", category="quality", status="skipped",
            reason="cross-session handoff needs a mutable adapter to seed fragments",
        )

    adapter.ensure_scope(_SCOPE)

    skip_reason = _supports_source_tool(adapter, _SCOPE)
    if skip_reason is not None:
        return ScenarioResult(
            name="cross_session_handoff",
            category="quality",
            status="skipped",
            reason=skip_reason,
        )

    # ---- Setup: tool A writes ------------------------------------------
    all_frags = fragments()
    chosen = [f for f in all_frags if (f.get("content") or "").strip()][:_N_FRAGMENTS]

    label_to_adapter_id: dict[str, str] = {}
    for f in chosen:
        adapter_id = adapter.remember(
            f["content"],
            type=f["type"],
            scope=_SCOPE,
            tags=f.get("tags", []),
            source_tool=_WRITE_TOOL,
        )
        label_to_adapter_id[f["id"]] = adapter_id

    # ---- Build the query list (simulating tool B's session) ------------
    by_expected_top: dict[str, dict[str, Any]] = {}
    for q in labeled_queries():
        top = q.get("expected_top")
        if top and top not in by_expected_top:
            by_expected_top[top] = q

    # ---- Query: tool B asks --------------------------------------------
    misses: list[dict[str, Any]] = []
    durations_ms: list[float] = []
    successes = 0
    total = len(chosen)

    for f in chosen:
        query = _query_for_fragment(f, by_expected_top)
        t0 = time.perf_counter()
        results = adapter.recall(
            query, scope=_SCOPE, limit=3, source_tool=_QUERY_TOOL,
        )
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

        expected_id = label_to_adapter_id[f["id"]]
        top_3_ids = [r.id for r in results]
        if expected_id in top_3_ids:
            successes += 1
        else:
            misses.append({
                "fragment_id": f["id"],
                "query": query,
                "top_3_ids": top_3_ids,
            })

    handoff_rate = successes / total if total else 0.0
    p50_latency_ms = (
        sorted(durations_ms)[len(durations_ms) // 2] if durations_ms else 0.0
    )

    if handoff_rate >= 0.7:
        status, reason = "pass", ""
    elif handoff_rate >= 0.5:
        status, reason = "warn", (
            f"handoff_rate={handoff_rate:.2f} below 0.7 target"
        )
    else:
        status, reason = "fail", (
            f"handoff_rate={handoff_rate:.2f} below 0.5 floor — "
            "tool A's writes are not reaching tool B's recalls"
        )

    return ScenarioResult(
        name="cross_session_handoff",
        category="quality",
        status=status,
        metrics={
            "handoff_rate": float(handoff_rate),
            "successes": float(successes),
            "total": float(total),
            "p50_latency_ms": float(p50_latency_ms),
        },
        reason=reason,
        notes={"misses": misses},
    )
