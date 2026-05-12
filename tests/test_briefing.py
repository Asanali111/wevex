"""Tests for the `project_briefing` MCP tool and `/v1/briefing` endpoint.

The briefing is a single-call project-state snapshot — fragment counts by
type, recent decisions, daemon health, recommended next action. Covered:
  * The storage helper `count_fragments_by_type` (single GROUP BY query).
  * The pure `build_briefing(storage, scope)` builder shape.
  * The recent-decisions cap at 5.
  * The three `next_recommended_action` heuristic branches.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from skein.models import FragmentCreate
from skein.storage import Storage


# ---------------------------------------------------------------------------
# count_fragments_by_type — single GROUP BY query
# ---------------------------------------------------------------------------

def test_count_fragments_by_type(seeded_storage: Storage) -> None:
    """Mixed-type seed → returns a dict keyed by type with the right counts."""
    st = seeded_storage
    scope_id = st._test_scope.id
    user_id = st._test_user.id

    # 3 decisions, 2 facts, 1 observation, 0 of everything else
    for content in ("d1", "d2", "d3"):
        st.create_fragment(FragmentCreate(
            type="decision", content=content,
            scope_id=scope_id, owner_id=user_id,
        ))
    for content in ("f1", "f2"):
        st.create_fragment(FragmentCreate(
            type="fact", content=content,
            scope_id=scope_id, owner_id=user_id,
        ))
    st.create_fragment(FragmentCreate(
        type="observation", content="o1",
        scope_id=scope_id, owner_id=user_id,
    ))

    counts = st.count_fragments_by_type(scope_id)
    assert counts == {"decision": 3, "fact": 2, "observation": 1}


def test_count_fragments_by_type_excludes_stale_by_default(seeded_storage: Storage) -> None:
    """Soft-deleted fragments don't count unless include_stale=True."""
    st = seeded_storage
    scope_id = st._test_scope.id
    user_id = st._test_user.id

    f1 = st.create_fragment(FragmentCreate(
        type="decision", content="alive",
        scope_id=scope_id, owner_id=user_id,
    ))
    f2 = st.create_fragment(FragmentCreate(
        type="decision", content="stale",
        scope_id=scope_id, owner_id=user_id,
    ))
    # Mark the second one stale.
    st._conn.execute(
        "UPDATE fragments SET is_stale = 1, stale_reason = 'test' WHERE id = ?",
        (f2.id,),
    )

    assert st.count_fragments_by_type(scope_id) == {"decision": 1}
    assert st.count_fragments_by_type(scope_id, include_stale=True) == {"decision": 2}


# ---------------------------------------------------------------------------
# build_briefing — pure function, shape contract
# ---------------------------------------------------------------------------

def test_project_briefing_shape(seeded_storage: Storage) -> None:
    """All top-level keys present with the right types."""
    from skein.mcp import build_briefing
    st = seeded_storage
    scope_id = st._test_scope.id
    user_id = st._test_user.id

    st.create_fragment(FragmentCreate(
        type="decision", content="picked SQLite over PostgreSQL",
        scope_id=scope_id, owner_id=user_id,
    ))
    st.create_fragment(FragmentCreate(
        type="fact", content="DB is at ~/.config/skein/skein.db",
        scope_id=scope_id, owner_id=user_id,
    ))

    out = build_briefing(st, "project:test")

    assert out["scope"] == "project:test"
    assert isinstance(out["fragment_counts"], dict)
    # Padded with all known types, even zero ones.
    for t in ("decision", "fact", "observation", "preference"):
        assert t in out["fragment_counts"]
    assert out["fragment_counts"]["decision"] == 1
    assert out["fragment_counts"]["fact"] == 1
    assert out["fragment_counts"]["preference"] == 0

    assert out["fragment_total"] == 2
    assert isinstance(out["chunks_total"], int)
    assert isinstance(out["recent_decisions"], list)
    assert out["active_inbox_count"] == 0
    assert isinstance(out["embedding_provider"], str)

    assert isinstance(out["daemon"], dict)
    assert out["daemon"]["version"] == "0.1.0"
    assert isinstance(out["daemon"]["uptime_seconds"], int)
    assert "db_path" in out["daemon"]

    assert isinstance(out["next_recommended_action"], str)
    assert out["next_recommended_action"]


def test_briefing_unknown_scope_returns_zeros(seeded_storage: Storage) -> None:
    """Calling briefing on a scope that doesn't exist yet returns zeros, not
    an error — the MCP tool should be safe to invoke from any cwd."""
    from skein.mcp import build_briefing
    out = build_briefing(seeded_storage, "project:does-not-exist")
    assert out["fragment_total"] == 0
    assert out["fragment_counts"]["decision"] == 0
    assert out["recent_decisions"] == []
    assert out["active_inbox_count"] == 0


# ---------------------------------------------------------------------------
# recent_decisions capped at 5
# ---------------------------------------------------------------------------

def test_briefing_recent_decisions_capped_at_5(seeded_storage: Storage) -> None:
    """Seeding 10 decisions → recent_decisions has exactly 5 entries."""
    from skein.mcp import build_briefing
    st = seeded_storage
    scope_id = st._test_scope.id
    user_id = st._test_user.id

    for i in range(10):
        st.create_fragment(FragmentCreate(
            type="decision", content=f"decision-{i:02d}",
            scope_id=scope_id, owner_id=user_id,
        ))

    out = build_briefing(st, "project:test")
    assert len(out["recent_decisions"]) == 5
    # The id_short field should be 8 chars (uuid[:8]).
    for d in out["recent_decisions"]:
        assert "id_short" in d
        assert len(d["id_short"]) == 8
        assert "content_first_line" in d
        assert "created_at" in d


# ---------------------------------------------------------------------------
# next_recommended_action — heuristic branches
# ---------------------------------------------------------------------------

def test_briefing_next_recommended_action_empty(seeded_storage: Storage) -> None:
    """Empty project → 'bootstrapping memory' message (< 10 decisions branch)."""
    from skein.mcp import build_briefing
    out = build_briefing(seeded_storage, "project:test")
    assert "bootstrapping memory" in out["next_recommended_action"]


def test_briefing_next_recommended_action_healthy(seeded_storage: Storage) -> None:
    """≥ 10 decisions and zero pending inbox → healthy message."""
    from skein.mcp import build_briefing
    st = seeded_storage
    scope_id = st._test_scope.id
    user_id = st._test_user.id

    for i in range(10):
        st.create_fragment(FragmentCreate(
            type="decision", content=f"decision-{i}",
            scope_id=scope_id, owner_id=user_id,
        ))

    out = build_briefing(st, "project:test")
    assert "healthy" in out["next_recommended_action"].lower()


def test_briefing_next_recommended_action_inbox_pending(
    seeded_storage: Storage,
) -> None:
    """Pending inbox candidates take priority over the other heuristics."""
    from skein.mcp import build_briefing
    st = seeded_storage
    scope_id = st._test_scope.id

    # Even with plenty of decisions, the inbox notice should win.
    user_id = st._test_user.id
    for i in range(12):
        st.create_fragment(FragmentCreate(
            type="decision", content=f"decision-{i}",
            scope_id=scope_id, owner_id=user_id,
        ))
    st.add_extraction_candidate(
        scope_id=scope_id, content="something pending",
        type="fact", territory=None, tags=[], confidence=0.5,
        source_tool="test-tool",
    )

    out = build_briefing(st, "project:test")
    msg = out["next_recommended_action"]
    assert "Review 1 pending fragments" in msg
    assert "skein inbox" in msg


# ---------------------------------------------------------------------------
# MCP wire integration + REST endpoint
# ---------------------------------------------------------------------------

def test_project_briefing_listed_in_tools(client: TestClient) -> None:
    """`tools/list` should expose `project_briefing`."""
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/list",
    })
    tools = resp.json()["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "project_briefing" in names
    # Sanity: description names the value prop, not just the action.
    briefing_tool = next(t for t in tools if t["name"] == "project_briefing")
    desc = briefing_tool["description"]
    assert "ONE call" in desc or "one call" in desc.lower()


def test_project_briefing_tool_call_returns_json_text(client: TestClient) -> None:
    """`tools/call project_briefing` returns the dict as JSON in the text content."""
    # Make sure the briefing scope exists.
    client.post("/v1/scopes", json={
        "handle": "project:briefing-mcp", "type": "project",
        "name": "Briefing MCP", "owner_id": "x",
    })
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {
            "name": "project_briefing",
            "arguments": {"scope": "project:briefing-mcp"},
        },
    })
    text = resp.json()["result"]["content"][0]["text"]
    payload = json.loads(text)
    assert payload["scope"] == "project:briefing-mcp"
    assert "fragment_counts" in payload
    assert "next_recommended_action" in payload


def test_briefing_http_endpoint(client: TestClient) -> None:
    """`GET /v1/briefing?scope=…` returns the briefing dict directly."""
    client.post("/v1/scopes", json={
        "handle": "project:briefing-http", "type": "project",
        "name": "Briefing HTTP", "owner_id": "x",
    })
    resp = client.get("/v1/briefing", params={"scope": "project:briefing-http"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["scope"] == "project:briefing-http"
    assert data["fragment_total"] == 0
    assert isinstance(data["daemon"], dict)
