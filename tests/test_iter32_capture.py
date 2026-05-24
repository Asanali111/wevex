"""Iter 32 antibodies — auto-capture + relevance marker + note() tool.

These tests pin the iter-32 structural fixes for "LLMs forget to write
to Skein":
  - git_watcher walks ``--branches`` (catches feat/* and experiment/*)
  - non-conv commits with substantive bodies bump to 0.90 (auto-promote)
  - transcript_watcher smart-only filter keeps only ≥0.90 patterns
  - recall response carries a ``[skein:relevance=…]`` marker (extension
    reads this to skip injection on low/none and stop wasting tokens)
  - note() one-arg MCP tool auto-classifies into a fragment type
"""
from __future__ import annotations

import subprocess
import textwrap

import pytest


# ---------------------------------------------------------------------------
# WS1 — git_watcher confidence + --branches
# ---------------------------------------------------------------------------


class TestGitWatcherConfidence:
    """commit_to_fact rules: conv → 0.92, non-conv long → 0.90,
    non-conv short → 0.75. AUTO_PROMOTE_THRESHOLD is 0.90."""

    def _make_commit(self, subject: str, body: str = ""):
        from skein.git_watcher import GitCommit
        return GitCommit(
            sha="abc1234567" + "0" * 30,
            author_name="x", author_email="x@example",
            timestamp="2026-05-21T00:00:00",
            subject=subject, body=body,
        )

    def test_conventional_commit_auto_promotes(self):
        from skein.git_watcher import commit_to_fact
        from skein.scanner import AUTO_PROMOTE_THRESHOLD
        c = self._make_commit("feat(daemon): add the thing")
        fact = commit_to_fact(c)
        assert fact.confidence >= AUTO_PROMOTE_THRESHOLD

    def test_non_conv_with_long_body_auto_promotes(self):
        from skein.git_watcher import commit_to_fact
        from skein.scanner import AUTO_PROMOTE_THRESHOLD
        long_body = "x" * 350
        c = self._make_commit("ship something big", body=long_body)
        fact = commit_to_fact(c)
        assert fact.confidence >= AUTO_PROMOTE_THRESHOLD, (
            "non-conv with body >=300 chars must auto-promote — iter 32 "
            "structural fix so iter-recap-style commits land as fragments "
            "instead of sitting in inbox"
        )

    def test_non_conv_short_stays_in_inbox(self):
        from skein.git_watcher import commit_to_fact
        from skein.scanner import AUTO_PROMOTE_THRESHOLD
        c = self._make_commit("quick fix", body="oops")
        fact = commit_to_fact(c)
        assert fact.confidence < AUTO_PROMOTE_THRESHOLD


class TestGitWatcherBranchesArg:
    """``read_commits_since`` must pass ``--branches`` to git so commits on
    feat/* and experiment/* branches reach Skein without first being merged
    to main. Iter 28-31 sat invisible for exactly this reason."""

    def test_read_commits_since_uses_branches_arg(self, tmp_path, monkeypatch):
        from skein import git_watcher
        captured: dict = {}

        def fake_run(args, capture_output=True, text=True, timeout=None):
            captured["args"] = list(args)
            class R:
                returncode = 0
                stdout = ""
            return R()

        # Make the .git existence check pass
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(git_watcher.subprocess, "run", fake_run)
        git_watcher.read_commits_since(tmp_path)
        assert "--branches" in captured["args"]

    def test_since_sha_passes_exclude_syntax(self, tmp_path, monkeypatch):
        from skein import git_watcher
        captured: dict = {}

        def fake_run(args, capture_output=True, text=True, timeout=None):
            captured["args"] = list(args)
            class R:
                returncode = 0
                stdout = ""
            return R()

        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(git_watcher.subprocess, "run", fake_run)
        git_watcher.read_commits_since(tmp_path, since_sha="deadbeef")
        # ``^<sha>`` excludes commits reachable from sha across all branches
        assert "^deadbeef" in captured["args"]
        # NOT the old "<sha>..HEAD" form (HEAD-only)
        assert not any(".." in a for a in captured["args"])


# ---------------------------------------------------------------------------
# WS1b — transcript_watcher smart-only filter
# ---------------------------------------------------------------------------


class TestTranscriptSmartOnly:
    """The smart-only filter must let ≥0.90 patterns through and reject the
    legacy 0.62–0.86 patterns that have historically polluted the inbox."""

    def test_iter_shipped_matches_under_smart_only(self):
        from skein.transcript_watcher import extract_from_text
        facts = extract_from_text(
            "Iter 31 SHIPPED — snippet rendering + 30s cache + dedupe.",
            role="assistant", smart_only=True,
        )
        assert facts, "smart-only must catch 'Iter N SHIPPED — …' sentences"
        assert any(f.confidence >= 0.90 for f in facts)

    def test_decided_to_matches_under_smart_only(self):
        from skein.transcript_watcher import extract_from_text
        facts = extract_from_text(
            "Decided to drop the old GeminiEmbeddingProvider.",
            role="assistant", smart_only=True,
        )
        assert facts
        assert all(f.confidence >= 0.90 for f in facts)

    def test_loose_pattern_filtered_under_smart_only(self):
        from skein.transcript_watcher import extract_from_text
        facts = extract_from_text(
            "i prefer flat directory structures",
            role="user", smart_only=True,
        )
        assert not facts, (
            "0.72-conf 'I prefer X' pattern must NOT fire under smart_only — "
            "that's the kind of inbox noise iter 32 is trying to suppress"
        )

    def test_loose_pattern_still_works_when_smart_only_false(self):
        from skein.transcript_watcher import extract_from_text
        facts = extract_from_text(
            "i prefer flat directory structures",
            role="user", smart_only=False,
        )
        assert facts, "loose mode must keep legacy behavior intact"


# ---------------------------------------------------------------------------
# WS2 — recall response relevance marker
# ---------------------------------------------------------------------------


class TestRecallRelevanceMarker:
    """The recall handler MUST prefix the rendered text with a
    ``[skein:relevance=…]`` line so the browser extension can parse it and
    skip injection on low/none — the headline token-waste fix of iter 32.
    """

    def test_relevance_marker_appears_in_recall_response(
        self, authed_client, app,
    ):
        # Seed a fragment so recall returns ≥1 result.
        from skein.dependencies import get_storage
        from skein.models import (
            FragmentCreate, IdentityCreate, ScopeCreate,
        )
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:reltest", type="user", name="Rel Test",
        ))
        scope = storage.get_or_create_scope(ScopeCreate(
            handle="project:reltest", type="project",
            name="Rel Test", owner_id=owner.id,
        ))
        storage.create_fragment(
            FragmentCreate(
                content="Decided to use sqlite WAL mode for hot-path reads.",
                type="decision", scope_id=scope.id, owner_id=owner.id,
                created_by_tool="test",
            ),
        )

        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {
                    "query": "sqlite mode",
                    "scope": "project:reltest",
                    "limit": 3,
                },
            },
        })
        assert resp.status_code == 200, resp.text
        text = resp.json()["result"]["content"][0]["text"]
        first_line = text.splitlines()[0].strip()
        assert first_line.startswith("[skein:relevance="), (
            f"recall response must start with relevance marker. Got: {first_line!r}"
        )
        assert first_line.endswith("]")

    def test_relevance_none_when_scope_empty(self, authed_client, app):
        # An empty scope should produce relevance=none (or the low-signal
        # nudge path, which also includes the marker).
        from skein.dependencies import get_storage
        from skein.models import IdentityCreate, ScopeCreate
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:empty32", type="user", name="Empty",
        ))
        storage.get_or_create_scope(ScopeCreate(
            handle="project:empty32", type="project",
            name="Empty", owner_id=owner.id,
        ))
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {
                    "query": "what is the meaning of life",
                    "scope": "project:empty32",
                    "limit": 3,
                },
            },
        })
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        # Marker must be present at the top
        assert "[skein:relevance=" in text.splitlines()[0]


# ---------------------------------------------------------------------------
# WS4 — note() one-arg MCP tool
# ---------------------------------------------------------------------------


class TestNoteTool:
    """note(content) is the low-friction write tool. It must:
       - accept a single content arg
       - auto-classify type (decision / requirement / preference / fact)
       - persist a fragment that recall can find
    """

    def test_note_tool_listed(self, authed_client, app):
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        })
        assert resp.status_code == 200
        names = [t["name"] for t in resp.json()["result"]["tools"]]
        assert "note" in names

    def test_note_classifies_decision(self, authed_client, app):
        from skein.dependencies import get_storage
        from skein.models import IdentityCreate, ScopeCreate
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:notetest", type="user", name="Note Test",
        ))
        storage.get_or_create_scope(ScopeCreate(
            handle="project:notetest", type="project",
            name="Note Test", owner_id=owner.id,
        ))
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "note",
                "arguments": {
                    "content": "Decided to use sqlite WAL mode for hot reads.",
                    "scope": "project:notetest",
                },
            },
        })
        assert resp.status_code == 200, resp.text
        text = resp.json()["result"]["content"][0]["text"]
        assert "decision" in text.lower()

    def test_note_persists_findable_fragment(self, authed_client, app):
        from skein.dependencies import get_storage
        from skein.models import IdentityCreate, ScopeCreate
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:notepersist", type="user", name="Note Persist",
        ))
        scope = storage.get_or_create_scope(ScopeCreate(
            handle="project:notepersist", type="project",
            name="Note Persist", owner_id=owner.id,
        ))
        # Write via the tool
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "note",
                "arguments": {
                    "content": "Iter 32 SHIPPED — note() tool plus relevance marker.",
                    "scope": "project:notepersist",
                },
            },
        })
        assert resp.status_code == 200
        # Verify it landed
        frags = storage.list_fragments(scope_id=scope.id, limit=10)
        assert any("Iter 32 SHIPPED" in (f.content or "") for f in frags)
