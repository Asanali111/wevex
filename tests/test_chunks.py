"""Tests for the codebase RAG layer: storage methods, ingest, search."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from skein.embeddings import HashEmbeddingProvider, vec_to_bytes
from skein.ingest import _chunk_text, ingest_directory
from skein.models import (
    ChunkCreate, ChunkSearchRequest, IdentityCreate, ScopeCreate,
)
from skein.retrieval import search_chunks


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def chunk_storage(seeded_storage):
    """A storage fixture with a seeded scope so chunk tests can attribute rows."""
    return seeded_storage


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Create a tiny fake codebase tree under tmp_path."""
    repo = tmp_path / "fake_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)
    (repo / "node_modules").mkdir(parents=True)  # excluded by default
    (repo / ".git").mkdir(parents=True)          # excluded

    # Python file
    (repo / "src" / "auth.py").write_text(textwrap.dedent("""
        \"\"\"Authentication module.\"\"\"

        def login(username, password):
            \"\"\"Validate credentials and issue a session token.\"\"\"
            if not username or not password:
                raise ValueError("missing credentials")
            return generate_token(username)


        def generate_token(username):
            \"\"\"Issue a short-lived JWT for the given user.\"\"\"
            return f"jwt:{username}"


        def logout(token):
            \"\"\"Revoke a session token.\"\"\"
            invalidate(token)
    """).strip())

    # TypeScript file
    (repo / "src" / "rate_limit.ts").write_text(textwrap.dedent("""
        export const RATE_LIMIT_PER_MINUTE = 1000;

        export function checkRateLimit(userId: string): boolean {
          const count = getCount(userId);
          return count < RATE_LIMIT_PER_MINUTE;
        }
    """).strip())

    # Markdown doc
    (repo / "docs" / "README.md").write_text(textwrap.dedent("""
        # Project

        This is a sample project for testing Skein's RAG ingestion.

        ## Authentication

        We use bearer tokens. Each request must include `Authorization: Bearer <token>`.

        ## Rate limiting

        The API enforces a per-user limit of 1000 requests per minute.
    """).strip())

    # File that should be skipped (binary-looking — too many invalid bytes)
    (repo / "binary.bin").write_bytes(b"\x00\x01\x02\x03" * 100)

    # File in node_modules (must be skipped via dir exclude)
    (repo / "node_modules" / "leftpad" / "index.js").parent.mkdir(parents=True, exist_ok=True)
    (repo / "node_modules" / "leftpad" / "index.js").write_text("export default function() {}")

    # File in .git (must be skipped)
    (repo / ".git" / "config").write_text("[core]\nrepositoryformatversion = 0")

    return repo


# ---------------------------------------------------------------------------
# Chunker (pure function)
# ---------------------------------------------------------------------------

class TestLineChunker:
    def test_short_text_one_chunk(self):
        text = "hello\nworld"
        out = _chunk_text(text, chunk_lines=80, overlap=10)
        assert len(out) == 1
        assert out[0]["content"] == "hello\nworld"
        assert out[0]["line_start"] == 1
        assert out[0]["line_end"] == 2

    def test_overlap(self):
        text = "\n".join(f"L{i}" for i in range(1, 51))  # 50 lines
        out = _chunk_text(text, chunk_lines=20, overlap=5)
        # Step = 20 - 5 = 15. Windows start at 0, 15, 30 — at i=30 the window
        # ends at line 50 (n), so the loop terminates. No need for a 4th chunk
        # because lines 31-50 are already covered. That's 3 chunks total.
        assert len(out) == 3
        assert out[0]["line_start"] == 1  and out[0]["line_end"] == 20
        assert out[1]["line_start"] == 16 and out[1]["line_end"] == 35
        assert out[2]["line_start"] == 31 and out[2]["line_end"] == 50

    def test_overlap_creates_overlap(self):
        # Verify adjacent chunks actually overlap by ``overlap`` lines.
        text = "\n".join(f"L{i}" for i in range(1, 31))  # 30 lines
        out = _chunk_text(text, chunk_lines=10, overlap=3)
        assert len(out) >= 2
        for a, b in zip(out, out[1:]):
            # b starts somewhere within a's range
            assert b["line_start"] <= a["line_end"]

    def test_overlap_clamped_to_chunk_size(self):
        text = "\n".join(f"L{i}" for i in range(1, 11))
        out = _chunk_text(text, chunk_lines=5, overlap=10)  # over-sized overlap
        assert len(out) >= 1
        assert all(c["line_end"] >= c["line_start"] for c in out)

    def test_empty_text(self):
        assert _chunk_text("") == []
        assert _chunk_text("   \n   ") == []


# ---------------------------------------------------------------------------
# Storage upsert + search
# ---------------------------------------------------------------------------

class TestChunkStorage:
    def test_upsert_inserts_then_idempotent(self, chunk_storage):
        scope = chunk_storage._test_scope
        data = ChunkCreate(
            scope_id=scope.id,
            source_root="myroot",
            source_path="src/foo.py",
            content="def foo(): return 42",
            line_start=1, line_end=1,
            language="python",
        )
        chunk1, status1 = chunk_storage.upsert_chunk(data, content_hash="hash-1")
        assert status1 == "inserted"

        # Same content_hash → no-op (unchanged)
        chunk2, status2 = chunk_storage.upsert_chunk(data, content_hash="hash-1")
        assert status2 == "unchanged"
        assert chunk2.id == chunk1.id

    def test_upsert_updates_on_hash_change(self, chunk_storage):
        scope = chunk_storage._test_scope
        data = ChunkCreate(
            scope_id=scope.id,
            source_root="myroot",
            source_path="src/foo.py",
            content="old",
            line_start=1, line_end=1,
            language="python",
        )
        chunk1, _ = chunk_storage.upsert_chunk(data, content_hash="hash-old")

        data.content = "new"
        chunk2, status = chunk_storage.upsert_chunk(data, content_hash="hash-new")
        assert status == "updated"
        assert chunk2.id == chunk1.id  # same row, content/hash refreshed
        assert chunk2.content == "new"
        assert chunk2.content_hash == "hash-new"

    def test_keyword_search(self, chunk_storage, provider):
        scope = chunk_storage._test_scope
        for content in [
            "Authentication uses bearer tokens",
            "Rate limit is 1000 req/min",
            "Database is Postgres with pgvector",
        ]:
            chunk_storage.upsert_chunk(
                ChunkCreate(
                    scope_id=scope.id, source_root="r",
                    source_path=f"f{hash(content) % 1000}.txt",
                    content=content, line_start=1, line_end=1,
                ),
                content_hash=str(hash(content)),
            )

        hits = chunk_storage.chunks_keyword_search(
            "bearer", [scope.id], limit=10,
        )
        assert hits
        # The first hit should be about bearer tokens
        chunk = chunk_storage.get_chunk(hits[0][0])
        assert "bearer" in chunk.content.lower()

    def test_vector_search(self, chunk_storage, provider):
        scope = chunk_storage._test_scope
        for i, content in enumerate([
            "Authentication uses bearer tokens for API requests.",
            "Rate limiting is 1000 requests per minute.",
            "Database storage uses Postgres with pgvector.",
        ]):
            emb = vec_to_bytes(provider.embed_one(content))
            chunk_storage.upsert_chunk(
                ChunkCreate(
                    scope_id=scope.id, source_root="r",
                    source_path=f"f{i}.txt",
                    content=content, line_start=1, line_end=1,
                ),
                content_hash=str(hash(content)),
                embedding=emb,
            )

        q_vec = vec_to_bytes(provider.embed_one(
            "Authentication uses bearer tokens for API requests."
        ))
        hits = chunk_storage.chunks_vector_search(
            q_vec, [scope.id], limit=10, dimension=provider.dimension,
        )
        assert hits
        # Top hit should be the exact match (deterministic for hash provider)
        top = chunk_storage.get_chunk(hits[0][0])
        assert "bearer" in top.content.lower()

    def test_delete_by_root(self, chunk_storage):
        scope = chunk_storage._test_scope
        for root in ["r1", "r2", "r1"]:
            chunk_storage.upsert_chunk(
                ChunkCreate(
                    scope_id=scope.id, source_root=root,
                    source_path=f"f-{root}.txt",
                    content=f"content for {root}",
                    line_start=1, line_end=1,
                ),
                content_hash=f"h-{root}-{id(object())}",
            )

        n = chunk_storage.delete_chunks_by_root(scope.id, "r1")
        assert n >= 1
        remaining = chunk_storage.list_chunks(scope_id=scope.id)
        assert all(c.source_root != "r1" for c in remaining)

    def test_chunk_stats(self, chunk_storage):
        scope = chunk_storage._test_scope
        for i, (root, lang) in enumerate([
            ("r1", "python"), ("r1", "python"),
            ("r2", "typescript"), ("r2", None),
        ]):
            chunk_storage.upsert_chunk(
                ChunkCreate(
                    scope_id=scope.id, source_root=root,
                    source_path=f"f{i}.txt",
                    content=f"content {i}",
                    line_start=1, line_end=1,
                    language=lang,
                ),
                content_hash=f"h{i}",
            )
        s = chunk_storage.chunk_stats(scope_id=scope.id)
        assert s["total_chunks"] == 4
        assert s["total_files"] == 4
        assert s["by_language"].get("python") == 2
        assert s["by_root"].get("r1") == 2


# ---------------------------------------------------------------------------
# ingest_directory (end-to-end on tmp tree)
# ---------------------------------------------------------------------------

class TestIngestDirectory:
    def test_walks_and_chunks(self, fake_repo, chunk_storage, provider):
        scope = chunk_storage._test_scope
        stats = ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
        )
        assert stats.files_seen >= 3   # auth.py, rate_limit.ts, README.md
        assert stats.files_ingested >= 3
        assert stats.chunks_inserted >= 3
        assert stats.errors == []

        # Confirm node_modules and .git are NOT ingested
        all_chunks = chunk_storage.list_chunks(scope_id=scope.id, limit=200)
        paths = {c.source_path for c in all_chunks}
        assert not any("node_modules" in p for p in paths)
        assert not any(".git" in p for p in paths)
        # Binary file skipped
        assert not any("binary.bin" in p for p in paths)

        # Languages detected
        langs = {c.language for c in all_chunks}
        assert "python" in langs
        assert "typescript" in langs
        assert "markdown" in langs

    def test_incremental_skips_unchanged(self, fake_repo, chunk_storage, provider):
        scope = chunk_storage._test_scope
        ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
        )
        before = chunk_storage.count_chunks(scope_id=scope.id)
        stats2 = ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
        )
        after = chunk_storage.count_chunks(scope_id=scope.id)
        assert before == after
        assert stats2.chunks_inserted == 0
        assert stats2.chunks_updated == 0
        assert stats2.chunks_unchanged == before

    def test_prune_missing(self, fake_repo, chunk_storage, provider):
        scope = chunk_storage._test_scope
        ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
        )
        # Delete a file from disk
        (fake_repo / "src" / "auth.py").unlink()

        stats = ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
            prune_missing=True,
        )
        assert stats.chunks_pruned >= 1

        remaining = chunk_storage.list_chunks(
            scope_id=scope.id, source_root="fake_repo", limit=200,
        )
        assert not any(c.source_path == "src/auth.py" for c in remaining)

    def test_dry_run_writes_nothing(self, fake_repo, chunk_storage, provider):
        scope = chunk_storage._test_scope
        before = chunk_storage.count_chunks(scope_id=scope.id)
        ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
            dry_run=True,
        )
        after = chunk_storage.count_chunks(scope_id=scope.id)
        assert before == after

    def test_include_filter(self, fake_repo, chunk_storage, provider):
        scope = chunk_storage._test_scope
        # Only Python files
        ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
            include_exts=[".py"],
        )
        chunks = chunk_storage.list_chunks(scope_id=scope.id, limit=200)
        assert chunks
        for c in chunks:
            assert c.source_path.endswith(".py")


# ---------------------------------------------------------------------------
# search_chunks (hybrid retrieval)
# ---------------------------------------------------------------------------

class TestSearchChunks:
    def test_finds_authentication_chunk(self, fake_repo, chunk_storage, provider):
        scope = chunk_storage._test_scope
        ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
        )
        req = ChunkSearchRequest(
            query="authentication and bearer tokens",
            scope=scope.handle,
            limit=5,
        )
        resp = search_chunks(req, chunk_storage, provider)
        assert resp.total > 0
        # Top hit must mention auth or bearer
        top = resp.results[0].chunk.content.lower()
        assert any(w in top for w in ("auth", "bearer", "token", "jwt"))

    def test_language_filter(self, fake_repo, chunk_storage, provider):
        scope = chunk_storage._test_scope
        ingest_directory(
            fake_repo, chunk_storage, provider,
            scope_id=scope.id, source_root="fake_repo",
        )
        req = ChunkSearchRequest(
            query="rate limit",
            scope=scope.handle,
            languages=["typescript"],
            limit=5,
        )
        resp = search_chunks(req, chunk_storage, provider)
        if resp.results:
            for r in resp.results:
                assert r.chunk.language == "typescript"

    def test_no_scope_returns_empty(self, chunk_storage, provider):
        req = ChunkSearchRequest(
            query="anything", scope="project:does-not-exist", limit=5,
        )
        resp = search_chunks(req, chunk_storage, provider)
        assert resp.total == 0
