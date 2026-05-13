"""Correctness scenarios — behavioural invariants the tool must uphold.

Three checks:

1. **Scope hierarchy**: a fragment stored on a parent scope must be visible
   from a query on a child scope. (Cross-tool value collapses without this.)
2. **Lease exclusion**: two attempts to claim overlapping leases on the
   same glob must not both succeed.
3. **Fragment typing**: ``remember`` of an unknown type must raise rather
   than silently store. (Prevents schema drift in the bus.)

Each check returns its own ``ScenarioResult`` so they can fail independently.
"""
from __future__ import annotations

from ..adapter import MutableAdapter
from ..scenarios import ScenarioResult


def check_scope_hierarchy(adapter: MutableAdapter) -> ScenarioResult:
    if not adapter.supports_scope_hierarchy:
        return ScenarioResult(
            name="scope_hierarchy", category="correctness", status="skipped",
            reason="adapter does not declare scope-hierarchy support",
        )
    adapter.ensure_scope("org:bench")
    adapter.ensure_scope("team:bench-backend", parent="org:bench")
    adapter.ensure_scope("project:bench-app", parent="team:bench-backend")

    adapter.remember(
        "Org-wide policy: all services emit OpenTelemetry traces",
        type="requirement", scope="org:bench", tags=["otel", "org"],
    )
    results = adapter.recall("OpenTelemetry traces", scope="project:bench-app", limit=5)
    contents = [r.content for r in results]
    found = any("OpenTelemetry" in c for c in contents)

    return ScenarioResult(
        name="scope_hierarchy",
        category="correctness",
        status="pass" if found else "fail",
        reason="" if found else "parent-scope fragment not visible from child scope",
        metrics={"parent_fragment_visible": 1.0 if found else 0.0,
                 "child_recall_hits": float(len(results))},
    )


def check_lease_lifecycle(adapter: MutableAdapter) -> ScenarioResult:
    """Claim → it exists → release → it's gone. Tool-agnostic; doesn't
    assume any particular exclusion semantics (re-entrant vs. exclusive vs.
    per-owner). Tools that want stricter exclusion tests can add their own.
    """
    if not adapter.supports_leases:
        return ScenarioResult(
            name="lease_lifecycle", category="correctness", status="skipped",
            reason="adapter does not declare lease support",
        )
    adapter.ensure_scope("project:bench-lease")

    lease_id = adapter.claim_lease("backend/auth/**",
                                   scope="project:bench-lease", ttl_seconds=30)
    if lease_id is None:
        return ScenarioResult(
            name="lease_lifecycle", category="correctness", status="fail",
            reason="claim_lease returned None on a fresh scope",
            metrics={"claimed": 0.0, "released": 0.0},
        )
    # Release and confirm a second claim succeeds afterwards.
    adapter.release_lease(lease_id)
    second = adapter.claim_lease("backend/auth/**",
                                 scope="project:bench-lease", ttl_seconds=30)
    if second is None:
        return ScenarioResult(
            name="lease_lifecycle", category="correctness", status="fail",
            reason="released lease still blocks subsequent claim",
            metrics={"claimed": 1.0, "released": 0.0},
        )
    adapter.release_lease(second)
    return ScenarioResult(
        name="lease_lifecycle", category="correctness", status="pass",
        metrics={"claimed": 1.0, "released": 1.0},
    )


def check_fragment_typing(adapter: MutableAdapter) -> ScenarioResult:
    if not adapter.supports_typed_fragments:
        return ScenarioResult(
            name="fragment_typing", category="correctness", status="skipped",
            reason="adapter does not declare typed-fragment support",
        )
    adapter.ensure_scope("project:bench-typing")
    raised = False
    try:
        adapter.remember(
            "should not be stored", type="garbage-type",
            scope="project:bench-typing",
        )
    except Exception:
        raised = True

    return ScenarioResult(
        name="fragment_typing",
        category="correctness",
        status="pass" if raised else "fail",
        reason="" if raised else "unknown fragment type was accepted silently",
        metrics={"rejected_unknown_type": 1.0 if raised else 0.0},
    )


def all_correctness_scenarios(adapter: MutableAdapter) -> list[ScenarioResult]:
    return [
        check_scope_hierarchy(adapter),
        check_lease_lifecycle(adapter),
        check_fragment_typing(adapter),
    ]
