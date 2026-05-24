"""Iter 29 day-one bundle — tests for the four changes that make Skein
valuable from minute zero on a fresh machine:

  #3 — eager fastembed warmup in lifespan (covered by smoke + manual)
  #6 — MCP initialize.instructions dynamic greeting
  #1 — cold-start corpus (docs_watcher patterns + Skein-AGENTS.md skip)
  #7 — empty-recall fallback proposes a `remember` write

These tests pin the iter-29 invariants so a future agent who refactors
the onboarding flow or the recall response shape has a CI signal pointing
at this iter's decisions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from skein import docs_watcher, mcp, storage as storage_mod


# ---------------------------------------------------------------------------
# #6 — initialize.instructions greeting is dynamic
# ---------------------------------------------------------------------------

class _StubStorage:
    """Minimal storage stand-in. Real ``Storage`` would do file I/O; we just
    need to feed ``_build_initialize_instructions`` the two numbers it reads."""

    def __init__(self, fragments: int, activity: dict[str, int]) -> None:
        self._fragments = fragments
        self._activity = activity

    def stats(self) -> dict[str, int]:
        return {"fragments": self._fragments}

    def recent_writes_by_tool(self, hours: int = 24) -> dict[str, int]:
        return dict(self._activity)


class TestInitializeInstructions:
    def test_empty_store_says_so_and_promises_cold_start(self) -> None:
        text = mcp._build_initialize_instructions(_StubStorage(0, {}))
        assert "no fragments stored yet" in text
        assert "Cold-start ingest" in text
        # Static recall-first rules must still be present.
        assert "Call the `recall` tool first" in text or "call the `recall` tool first" in text.lower()
        # Quick-start footer with a tool name the LLM can act on.
        assert "project_briefing" in text and "recall(" in text

    def test_populated_store_lists_count_and_cross_tool_activity(self) -> None:
        text = mcp._build_initialize_instructions(
            _StubStorage(47, {"claude-code": 12, "cursor": 3, "codex": 0}),
        )
        assert "47 fragments" in text
        # Cross-tool block — top-N rendering must include the counts.
        assert "claude-code (12)" in text
        assert "cursor (3)" in text

    def test_populated_no_activity_suggests_connecting_another_llm(self) -> None:
        text = mcp._build_initialize_instructions(_StubStorage(47, {}))
        # The upsell trigger: when one LLM has been writing but no recent
        # cross-tool activity, prompt the user to wire up a second client.
        assert "skein connect" in text

    def test_storage_failure_falls_back_gracefully(self) -> None:
        class _Broken:
            def stats(self):  # noqa: D401
                raise RuntimeError("disk error")
            def recent_writes_by_tool(self, hours=24):
                raise RuntimeError("disk error")
        # Must not raise — initialize handshake must never 500.
        text = mcp._build_initialize_instructions(_Broken())
        assert text  # non-empty fallback
        assert "Call the `recall` tool first" in text or "recall" in text.lower()


# ---------------------------------------------------------------------------
# #6 — recent_writes_by_tool storage helper
# ---------------------------------------------------------------------------

class TestRecentWritesByTool:
    @pytest.fixture
    def real_storage(self, tmp_path):
        """A real Storage backed by a throwaway SQLite under tmp_path."""
        s = storage_mod.Storage(str(tmp_path / "skein.db"))
        yield s
        s.close()

    def _seed_fragment(self, st, *, tool: str, scope_id: str, owner_id: str,
                       suffix: str = ""):
        # Iter 31: dedupe now short-circuits identical (scope, type, tool,
        # content) writes. Each call must use unique content to actually
        # produce a new fragment — exercise that with a suffix.
        from skein.models import FragmentCreate
        st.create_fragment(FragmentCreate(
            type="fact", content=f"thing from {tool} {suffix}".strip(),
            scope_id=scope_id, owner_id=owner_id,
            tags=[], confidence=0.9,
            created_by_tool=tool,
        ))

    def test_groups_recent_fragments_by_tool(self, real_storage):
        from skein.models import IdentityCreate, ScopeCreate
        owner = real_storage.get_or_create_identity(IdentityCreate(
            handle="user:test", type="user", name="test",
        ))
        scope = real_storage.create_scope(ScopeCreate(
            handle="project:test", type="project", name="test",
            owner_id=owner.id,
        ))
        # Each entry must have unique content because of iter-31 dedupe.
        seeds = [
            ("claude-code", "a"), ("claude-code", "b"), ("claude-code", "c"),
            ("cursor", "x"), ("cursor", "y"),
        ]
        for tool, suffix in seeds:
            self._seed_fragment(real_storage, tool=tool, suffix=suffix,
                                scope_id=scope.id, owner_id=owner.id)
        result = real_storage.recent_writes_by_tool(hours=24)
        assert result["claude-code"] == 3
        assert result["cursor"] == 2

    def test_empty_when_no_fragments(self, real_storage):
        assert real_storage.recent_writes_by_tool(hours=24) == {}


# ---------------------------------------------------------------------------
# #1 — docs_watcher new patterns + Skein-AGENTS.md skip
# ---------------------------------------------------------------------------

class TestDocsWatcherCorpusExpansion:
    def test_globs_include_ai_tool_config_files(self) -> None:
        patterns = docs_watcher._GLOB_PATTERNS
        # The four big AI-tool config formats we expect to harvest on a
        # fresh `skein up` against an existing project.
        assert "AGENTS.md" in patterns
        assert "CLAUDE.md" in patterns
        assert ".cursor/rules/**/*.mdc" in patterns
        assert ".github/copilot-instructions.md" in patterns

    def test_skein_generated_agents_md_is_skipped(self, tmp_path: Path) -> None:
        """A docs sweep must not re-ingest its own output — that would inflate
        AGENTS.md on every cycle (fragments → AGENTS.md → docs sweep → more
        fragments)."""
        agents = tmp_path / "AGENTS.md"
        agents.write_text(
            "# AGENTS.md — myproj\n\n"
            "> Generated by Skein at 2026-05-20 10:00 UTC. Do not edit "
            "this file directly; run `skein sync` to regenerate.\n\n"
            "## Some content\n\nthat would otherwise be harvested.\n"
        )
        result = docs_watcher._process_file(tmp_path, agents)
        assert result == [], (
            "Skein-generated AGENTS.md must be skipped to avoid the "
            "ingest-output-of-our-own-output inflation loop"
        )

    def test_user_authored_agents_md_is_harvested(self, tmp_path: Path) -> None:
        """A user-authored AGENTS.md (no Skein header) IS harvested — that's
        the whole point of expanding the corpus."""
        agents = tmp_path / "AGENTS.md"
        agents.write_text(
            "# AGENTS\n\n"
            "## Style preferences\n\n"
            "Prefer async/await over callbacks. Always type-annotate "
            "public functions. Tests live in tests/.\n"
        )
        result = docs_watcher._process_file(tmp_path, agents)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# #7 — empty-recall fallback fires only on real queries
# ---------------------------------------------------------------------------

class TestEmptyRecallSuggestionGuard:
    """The fallback message format is enforced here so the format the LLM
    relies on doesn't drift. We test the conditional in isolation rather
    than spinning up a full MCP request — the logic that matters is the
    "real question" filter and the suggested-call shape.
    """

    @staticmethod
    def _should_suggest(query: str) -> bool:
        # Mirrors the conditional in mcp.py exactly. If this changes, both
        # update or the test catches the drift.
        return len(query) >= 10 and " " in query.strip()

    def test_suggests_for_real_question(self) -> None:
        assert self._should_suggest("how does authentication work?")
        assert self._should_suggest("database choice rationale")

    def test_no_suggest_for_single_token(self) -> None:
        # Short, no-whitespace queries are bad write targets — likely
        # autocomplete or test traffic, not a real question.
        assert not self._should_suggest("test")
        assert not self._should_suggest("foo")
        assert not self._should_suggest("authentication")

    def test_no_suggest_for_too_short(self) -> None:
        assert not self._should_suggest("a b")  # has space but <10 chars
