"""Regression tests for iteration 11: AI-quality + speed fixes.

Each test pins a specific bug fix from the iteration so future refactors
don't silently re-introduce noise into the prompt-injection path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from skein import hooks as hooks_mod
from skein.agents_md import _extract_user_block, render_agents_md
from skein.scope_resolver import _is_non_project_dir, auto_detect_scope


# Iter 27: the POSIX `Path("/")` and `Path("/tmp")` cases don't translate
# to Windows — `Path("/")` becomes `WindowsPath('/')` (drive-relative,
# not a system root) and there is no `/tmp`. The non-project-dir guards
# in scope_resolver.py are derived from a POSIX `_SYSTEM_ROOTS` list;
# building the Windows equivalent (`C:\`, `C:\Windows`, …) is its own
# small project and not blocking for first-cut Windows support. Skip
# these three tests on Windows; the same code paths are exercised on
# Linux + macOS CI.
_skip_posix_only = pytest.mark.skipif(
    sys.platform.startswith("win") or os.name == "nt",
    reason="POSIX-only system-root list; Windows equivalent is follow-up work",
)


# ---------------------------------------------------------------------------
# Scope auto-detect — refuse $HOME-named scopes
# ---------------------------------------------------------------------------

class TestScopeGuard:
    def test_home_dir_returns_personal_scratch(self, monkeypatch):
        monkeypatch.setattr(Path, "home",
                            classmethod(lambda cls: Path("/Users/test")))
        # Pretend cwd is the home dir
        result = auto_detect_scope(Path("/Users/test"))
        assert result == "personal:scratch"

    @_skip_posix_only
    def test_root_returns_personal_scratch(self):
        assert auto_detect_scope(Path("/")) == "personal:scratch"

    @_skip_posix_only
    def test_tmp_returns_personal_scratch(self):
        assert auto_detect_scope(Path("/tmp")) == "personal:scratch"

    def test_real_project_dir_unchanged(self, tmp_path, monkeypatch):
        # tmp_path is not $HOME and not in the deny list → normal handle
        monkeypatch.setattr(Path, "home",
                            classmethod(lambda cls: Path("/Users/test")))
        result = auto_detect_scope(tmp_path)
        assert result.startswith("project:")
        assert "scratch" not in result

    @_skip_posix_only
    def test_is_non_project_dir_helper(self, monkeypatch):
        monkeypatch.setattr(Path, "home",
                            classmethod(lambda cls: Path("/Users/test")))
        assert _is_non_project_dir(Path("/Users/test"))
        assert _is_non_project_dir(Path("/"))
        assert _is_non_project_dir(Path("/Users/test/Documents"))
        assert not _is_non_project_dir(Path("/Users/test/Documents/myapp"))


# ---------------------------------------------------------------------------
# AGENTS.md regex — anchored markers
# ---------------------------------------------------------------------------

class TestUserBlockExtractor:
    def test_extracts_simple_block(self):
        text = (
            "Header\n"
            "<!-- @user -->\n"
            "user content here\n"
            "<!-- /@user -->\n"
            "Footer"
        )
        block = _extract_user_block(text)
        assert "user content here" in block

    def test_does_not_match_inline_marker(self):
        """The header documentation line contains a literal `<!-- @user -->`
        in backticks. The anchored regex must not match it."""
        text = (
            "# AGENTS.md\n"
            "> Add custom content in the `<!-- @user -->` block.\n"
            "\n"
            "## Body content\n"
            "Real body here.\n"
            "\n"
            "<!-- @user -->\n"
            "real user block\n"
            "<!-- /@user -->\n"
        )
        block = _extract_user_block(text)
        assert "real user block" in block
        # The body must NOT be in the extracted user block
        assert "Real body here" not in block
        assert "## Body content" not in block

    def test_idempotent_round_trip(self, tmp_path):
        """Run sync 5× in a row; AGENTS.md should stay constant size."""
        from skein.storage import Storage
        from skein.models import IdentityCreate, ScopeCreate

        s = Storage(str(tmp_path / "test.db"))
        owner = s.create_identity(IdentityCreate(
            handle="user:t", type="user", name="t",
        ))
        s.create_scope(ScopeCreate(
            handle="project:agents-md-test", type="project",
            name="agents-md-test", owner_id=owner.id,
        ))

        prev_text = ""
        for i in range(5):
            text = render_agents_md(
                "project:agents-md-test", s,
                daemon_url="http://127.0.0.1:8765",
                existing_content=prev_text,
            )
            prev_text = text

        s.close()
        # After 5 round-trips, exactly one Skein-context-bus header
        assert prev_text.count("## Skein context bus") == 1
        # The header documentation line legitimately contains `<!-- @user -->`
        # in backticks; we want exactly one un-backticked marker pair (the
        # real block).
        import re as _re
        unbackticked_open = _re.findall(
            r"^<!--\s*@user\s*-->", prev_text, _re.MULTILINE,
        )
        unbackticked_close = _re.findall(
            r"^<!--\s*/@user\s*-->", prev_text, _re.MULTILINE,
        )
        assert len(unbackticked_open) == 1
        assert len(unbackticked_close) == 1


# ---------------------------------------------------------------------------
# Hook injection: SIGNAL_TYPES filter + dedupe + threshold
# ---------------------------------------------------------------------------

class TestHookConstants:
    def test_signal_types_excludes_observation(self):
        assert "observation" not in hooks_mod.SIGNAL_TYPES
        assert "conversation" not in hooks_mod.SIGNAL_TYPES
        assert "decision" in hooks_mod.SIGNAL_TYPES
        assert "requirement" in hooks_mod.SIGNAL_TYPES

    def test_min_inject_score_is_above_noise_floor(self):
        """0.005 was too lenient — 0.016 noise hits got through. The new
        threshold must reject anything below 0.025."""
        assert hooks_mod.MIN_INJECT_SCORE >= 0.025


class TestRenderGrouped:
    def test_groups_by_type(self):
        from skein.models import Fragment

        frags = [
            Fragment(
                id="1", type="decision", content="Use Redis", scope_id="s",
                owner_id="o", confidence=1.0, version=1, ttl_seconds=None,
                expires_at=None, permanent=True, is_stale=False,
                stale_reason=None, tags=[], territory=None,
                source_commit_id=None, metadata={},
                created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
            ),
            Fragment(
                id="2", type="requirement", content="API rate limit 1000/min",
                scope_id="s", owner_id="o", confidence=1.0, version=1,
                ttl_seconds=None, expires_at=None, permanent=True, is_stale=False,
                stale_reason=None, tags=[], territory=None,
                source_commit_id=None, metadata={},
                created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
            ),
        ]
        text = hooks_mod._render_grouped(
            "project:test", frags, header="Test",
        )
        # Requirements come before decisions per SECTION_ORDER
        req_idx = text.find("Requirements")
        dec_idx = text.find("Decisions")
        assert req_idx > -1 and dec_idx > -1
        assert req_idx < dec_idx

    def test_renders_no_observations_section_when_absent(self):
        from skein.models import Fragment
        frags = [
            Fragment(
                id="1", type="decision", content="A decision", scope_id="s",
                owner_id="o", confidence=1.0, version=1, ttl_seconds=None,
                expires_at=None, permanent=True, is_stale=False,
                stale_reason=None, tags=[], territory=None,
                source_commit_id=None, metadata={},
                created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
            ),
        ]
        text = hooks_mod._render_grouped(
            "project:test", frags, header="Test",
        )
        assert "Decisions" in text
        assert "Observations" not in text


# ---------------------------------------------------------------------------
# Storage perf helpers
# ---------------------------------------------------------------------------

class TestCountFragmentsInScope:
    def test_returns_zero_for_new_scope(self, tmp_path):
        from skein.storage import Storage
        from skein.models import IdentityCreate, ScopeCreate

        s = Storage(str(tmp_path / "test.db"))
        owner = s.create_identity(IdentityCreate(
            handle="user:t", type="user", name="t",
        ))
        scope = s.create_scope(ScopeCreate(
            handle="project:empty", type="project",
            name="empty", owner_id=owner.id,
        ))
        assert s.count_fragments_in_scope(scope.id) == 0
        s.close()


class TestIngestFastPath:
    """Iteration 12: bulk-precheck hashes; skip embed+upsert for unchanged."""

    def _build_repo(self, tmp_path: Path, n_files: int = 5):
        repo = tmp_path / "repo"
        repo.mkdir()
        for i in range(n_files):
            (repo / f"mod_{i}.py").write_text(
                "\n".join(f"def f_{i}_{j}(): pass" for j in range(40))
            )
        return repo

    def _build_storage_with_scope(self, tmp_path: Path):
        from skein.storage import Storage
        from skein.models import IdentityCreate, ScopeCreate
        s = Storage(str(tmp_path / "ingest.db"))
        owner = s.create_identity(IdentityCreate(
            handle="user:i", type="user", name="i",
        ))
        scope = s.create_scope(ScopeCreate(
            handle="project:ingest-bench", type="project",
            name="bench", owner_id=owner.id,
        ))
        return s, scope

    def test_unchanged_chunks_skip_embed(self, tmp_path):
        """If every chunk in a batch is unchanged, the provider's embed()
        must NOT be called. This is the 20–50× win on re-ingest with
        Gemini."""
        from skein.ingest import ingest_directory
        from skein.embeddings import HashEmbeddingProvider

        repo = self._build_repo(tmp_path)
        storage, scope = self._build_storage_with_scope(tmp_path)

        # First pass: populate
        ingest_directory(
            repo, storage, HashEmbeddingProvider(),
            scope_id=scope.id, source_root="bench",
        )

        # Second pass: instrument the provider — embed() must not be called.
        class SpyProvider(HashEmbeddingProvider):
            calls = 0

            def embed(self, texts):
                SpyProvider.calls += 1
                return super().embed(texts)

        spy = SpyProvider()
        stats = ingest_directory(
            repo, storage, spy,
            scope_id=scope.id, source_root="bench",
        )
        assert stats.chunks_inserted == 0
        assert stats.chunks_updated == 0
        assert stats.chunks_unchanged > 0
        assert SpyProvider.calls == 0, (
            "embed() was called for an all-unchanged batch — fast-path broken"
        )
        storage.close()

    def test_changed_chunk_triggers_embed_only_for_that_chunk(self, tmp_path):
        """One file changed → only its chunks should reach embed()."""
        from skein.ingest import ingest_directory
        from skein.embeddings import HashEmbeddingProvider

        repo = self._build_repo(tmp_path, n_files=5)
        storage, scope = self._build_storage_with_scope(tmp_path)

        ingest_directory(
            repo, storage, HashEmbeddingProvider(),
            scope_id=scope.id, source_root="bench",
        )

        # Modify one file's content
        (repo / "mod_2.py").write_text("# new content\n" + "pass\n" * 40)

        embedded_texts: list = []

        class SpyProvider(HashEmbeddingProvider):
            def embed(self, texts):
                embedded_texts.extend(texts)
                return super().embed(texts)

        stats = ingest_directory(
            repo, storage, SpyProvider(),
            scope_id=scope.id, source_root="bench",
        )
        assert stats.chunks_updated + stats.chunks_inserted > 0
        # The only texts passed to embed() should be from mod_2.py
        assert embedded_texts, "no embed() calls at all — unexpected"
        for t in embedded_texts:
            assert "# new content" in t or "pass" in t
        storage.close()


class TestDaemonBootFastPath:
    """Iteration 12: daemon status uses cached backend + stdlib urllib so
    `skein up` doesn't pay the 800 ms launchctl + 1100 ms httpx cold-start
    cost on every run."""

    def test_check_health_uses_stdlib_not_httpx(self):
        """If we ever re-introduce httpx in _check_health, every `skein up`
        gains ~1.1 s back. Pin it."""
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "skein" / "daemon.py"
        text = src.read_text()
        # Find the _check_health body and assert urllib usage
        marker = "def _check_health"
        idx = text.find(marker)
        assert idx != -1
        # Look at the first ~30 lines of the function body
        body = text[idx:idx + 1500]
        assert "urllib.request" in body, (
            "_check_health must use stdlib urllib, not httpx"
        )
        assert "import httpx" not in body, (
            "_check_health must not import httpx (cold-import cost ~700 ms)"
        )

    def test_cached_backend_read_write_roundtrip(self, tmp_path, monkeypatch):
        from skein import daemon as d_mod
        monkeypatch.setattr(
            d_mod, "_BACKEND_CACHE_FILE", tmp_path / "backend",
        )
        # Pretend the plist exists so the validator passes
        monkeypatch.setattr(d_mod, "LAUNCHD_PLIST", tmp_path / "plist.fake")
        (tmp_path / "plist.fake").write_text("")

        assert d_mod._cached_backend() is None  # empty
        d_mod._write_cached_backend("launchd")
        assert d_mod._cached_backend() == "launchd"

    def test_cached_backend_invalidated_when_unit_missing(self, tmp_path, monkeypatch):
        """If the user uninstalls launchd unit out-of-band, cache should
        refuse to report it as backend."""
        from skein import daemon as d_mod
        monkeypatch.setattr(
            d_mod, "_BACKEND_CACHE_FILE", tmp_path / "backend",
        )
        monkeypatch.setattr(d_mod, "LAUNCHD_PLIST", tmp_path / "no.plist")
        d_mod._write_cached_backend("launchd")
        # Plist doesn't exist → cache returns None
        assert d_mod._cached_backend() is None


class TestPragmaTuning:
    """Verify the connection has the expected PRAGMA settings — these
    drive the speed budget for the daemon and watcher."""

    def test_journal_mode_wal(self, tmp_path):
        from skein.storage import Storage
        s = Storage(str(tmp_path / "test.db"))
        try:
            mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            s.close()

    def test_synchronous_normal(self, tmp_path):
        from skein.storage import Storage
        s = Storage(str(tmp_path / "test.db"))
        try:
            sync = s._conn.execute("PRAGMA synchronous").fetchone()[0]
            assert sync == 1  # NORMAL
        finally:
            s.close()

    def test_busy_timeout_set(self, tmp_path):
        from skein.storage import Storage
        s = Storage(str(tmp_path / "test.db"))
        try:
            t = s._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert t >= 30000
        finally:
            s.close()
