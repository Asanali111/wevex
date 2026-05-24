"""Iter 35 antibodies — recall→write outcome telemetry.

The daemon mints a recall_id on every MCP recall and lets the caller pass
it back via remember(from_recall=...) / note(from_recall=...). The pair is
recorded in recall_events + recall_links so the doctor "recall→write
rate" line (and, in a future iter, the value-decay loop) can learn from
outcomes instead of hit counts alone.

These tests pin the contract end-to-end:
  - storage records the event and the link
  - link_recall_to_fragment is a silent no-op for unknown recall_ids
  - recall_write_stats math is correct inside / outside the window
  - the MCP recall response carries the recall_id footer the LLM needs
  - MCP remember with from_recall actually creates the link
"""
from __future__ import annotations


class TestRecordRecallEvent:
    def test_record_then_query(self, seeded_storage):
        s = seeded_storage
        s.record_recall_event("abc123def456", "how does auth work", "project:test")
        row = s._conn.execute(
            "SELECT recall_id, query, scope_handle FROM recall_events WHERE recall_id = ?",
            ("abc123def456",),
        ).fetchone()
        assert row is not None
        assert row["recall_id"] == "abc123def456"
        assert row["query"] == "how does auth work"
        assert row["scope_handle"] == "project:test"

    def test_duplicate_recall_id_is_no_op(self, seeded_storage):
        s = seeded_storage
        s.record_recall_event("dup1", "first", "project:test")
        s.record_recall_event("dup1", "second-should-be-ignored", "project:test")
        rows = s._conn.execute(
            "SELECT query FROM recall_events WHERE recall_id = ?", ("dup1",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["query"] == "first"


class TestLinkRecallToFragment:
    def test_link_unknown_recall_id_returns_false(self, seeded_storage):
        from skein.models import FragmentCreate
        s = seeded_storage
        frag = s.create_fragment(FragmentCreate(
            content="test fact", type="fact",
            scope_id=s._test_scope.id, owner_id=s._test_user.id,
        ))
        assert s.link_recall_to_fragment("nonexistent-id", frag.id) is False

    def test_link_known_recall_id_returns_true_and_persists(self, seeded_storage):
        from skein.models import FragmentCreate
        s = seeded_storage
        s.record_recall_event("known1", "q", "project:test")
        frag = s.create_fragment(FragmentCreate(
            content="follow-up fact", type="fact",
            scope_id=s._test_scope.id, owner_id=s._test_user.id,
        ))
        assert s.link_recall_to_fragment("known1", frag.id) is True
        row = s._conn.execute(
            "SELECT recall_id, fragment_id FROM recall_links WHERE recall_id = ?",
            ("known1",),
        ).fetchone()
        assert row is not None
        assert row["fragment_id"] == frag.id


class TestRecallWriteStats:
    def test_empty_window_returns_zero_zero(self, seeded_storage):
        linked, total = seeded_storage.recall_write_stats(hours=24)
        assert (linked, total) == (0, 0)

    def test_unlinked_recalls_count_in_total_not_linked(self, seeded_storage):
        s = seeded_storage
        s.record_recall_event("r1", "q1", "project:test")
        s.record_recall_event("r2", "q2", "project:test")
        linked, total = s.recall_write_stats(hours=24)
        assert total == 2
        assert linked == 0

    def test_linked_recalls_count_in_both(self, seeded_storage):
        from skein.models import FragmentCreate
        s = seeded_storage
        s.record_recall_event("rA", "qA", "project:test")
        s.record_recall_event("rB", "qB", "project:test")
        frag = s.create_fragment(FragmentCreate(
            content="x", type="fact",
            scope_id=s._test_scope.id, owner_id=s._test_user.id,
        ))
        s.link_recall_to_fragment("rA", frag.id)
        linked, total = s.recall_write_stats(hours=24)
        assert total == 2
        assert linked == 1


class TestMCPRecallFooter:
    """The MCP recall response MUST include a `[skein:recall_id=...]`
    footer line so the LLM can pass the id back via from_recall."""

    def test_recall_response_includes_recall_id_footer(self, authed_client, app):
        from skein.dependencies import get_storage
        from skein.models import FragmentCreate, IdentityCreate, ScopeCreate
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:i35", type="user", name="iter35",
        ))
        scope = storage.get_or_create_scope(ScopeCreate(
            handle="project:i35", type="project",
            name="Iter 35", owner_id=owner.id,
        ))
        storage.create_fragment(FragmentCreate(
            content="Decided to track recall outcomes via explicit from_recall.",
            type="decision", scope_id=scope.id, owner_id=owner.id,
            created_by_tool="test",
        ))
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {
                    "query": "track recall outcomes",
                    "scope": "project:i35",
                    "limit": 3,
                },
            },
        })
        assert resp.status_code == 200, resp.text
        text = resp.json()["result"]["content"][0]["text"]
        assert "[skein:recall_id=" in text, (
            f"recall response must include recall_id footer. Got: {text!r}"
        )

    def test_recall_response_footer_present_when_empty(self, authed_client, app):
        from skein.dependencies import get_storage
        from skein.models import IdentityCreate, ScopeCreate
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:i35e", type="user", name="iter35-empty",
        ))
        storage.get_or_create_scope(ScopeCreate(
            handle="project:i35e", type="project",
            name="Iter 35 Empty", owner_id=owner.id,
        ))
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {
                    "query": "this matches nothing at all xyzqrs",
                    "scope": "project:i35e",
                    "limit": 3,
                },
            },
        })
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        assert "[skein:recall_id=" in text


class TestMCPRememberFromRecall:
    """Calling remember(from_recall=<id>) creates a row in recall_links."""

    def test_remember_with_from_recall_creates_link(self, authed_client, app):
        from skein.dependencies import get_storage
        from skein.models import IdentityCreate, ScopeCreate
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:i35r", type="user", name="iter35-remember",
        ))
        storage.get_or_create_scope(ScopeCreate(
            handle="project:i35r", type="project",
            name="Iter 35 Remember", owner_id=owner.id,
        ))
        # Recall first to mint a recall_id
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {"query": "any query at all here",
                              "scope": "project:i35r", "limit": 3},
            },
        })
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        # Parse the recall_id out of the footer
        import re
        m = re.search(r"\[skein:recall_id=([0-9a-f]+)\]", text)
        assert m, f"could not extract recall_id from response: {text!r}"
        recall_id = m.group(1)

        # Now remember with from_recall
        resp2 = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "remember",
                "arguments": {
                    "content": "After the recall, we decided to add telemetry.",
                    "type": "decision",
                    "scope": "project:i35r",
                    "from_recall": recall_id,
                },
            },
        })
        assert resp2.status_code == 200
        text2 = resp2.json()["result"]["content"][0]["text"]
        assert "linked to recall" in text2, (
            f"remember response should mention link: {text2!r}"
        )

        # And verify the link exists in storage
        row = storage._conn.execute(
            "SELECT fragment_id FROM recall_links WHERE recall_id = ?",
            (recall_id,),
        ).fetchone()
        assert row is not None

    def test_remember_with_bogus_from_recall_does_not_error(self, authed_client, app):
        from skein.dependencies import get_storage
        from skein.models import IdentityCreate, ScopeCreate
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:i35b", type="user", name="iter35-bogus",
        ))
        storage.get_or_create_scope(ScopeCreate(
            handle="project:i35b", type="project",
            name="Iter 35 Bogus", owner_id=owner.id,
        ))
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "remember",
                "arguments": {
                    "content": "test with bogus from_recall",
                    "type": "fact",
                    "scope": "project:i35b",
                    "from_recall": "does-not-exist",
                },
            },
        })
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        # No link should be added, but the call should NOT error
        assert "Stored fragment" in text
        assert "linked to recall" not in text
