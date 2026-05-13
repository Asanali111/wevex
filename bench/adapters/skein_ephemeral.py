"""Ephemeral Skein adapter — fresh in-process daemon backed by a temp SQLite DB.

Used by the pytest suite and by ``bench --ephemeral``. Talks directly to
``Storage`` / ``EmbeddingProvider`` so we don't pay HTTP overhead and so
scenarios can introspect state freely. Numbers are app-level, not
network-inclusive — that's why the live adapter exists separately.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..adapter import (
    CodeChunkResult,
    FragmentResult,
    HealthInfo,
    MutableAdapter,
)


class SkeinEphemeralAdapter(MutableAdapter):
    """In-process Skein backed by ``tempfile.mkdtemp`` SQLite + hash embeddings."""

    name = "skein-ephemeral"
    supports_typed_fragments = True
    supports_leases = True
    supports_code_search = True
    supports_scope_hierarchy = True
    supports_git_capture = True

    def __init__(self, *, embedding_provider: str = "hash"):
        self._embedding_provider_name = embedding_provider
        self._tmpdir: Path | None = None
        self._storage = None
        self._provider = None
        self._user_id: str | None = None
        self._scopes: dict[str, str] = {}  # handle -> id
        self._open()

    # ---- lifecycle ------------------------------------------------------

    def _open(self) -> None:
        from skein.embeddings import get_provider
        from skein.models import IdentityCreate
        from skein.storage import Storage

        self._tmpdir = Path(tempfile.mkdtemp(prefix="bench_skein_"))
        db_path = self._tmpdir / "skein.db"
        self._storage = Storage(str(db_path))
        self._provider = get_provider(self._embedding_provider_name)

        user = self._storage.create_identity(IdentityCreate(
            handle="user:bench", type="user", name="Benchmark Runner",
        ))
        self._user_id = user.id

    def close(self) -> None:
        if self._storage is not None:
            try:
                self._storage.close()
            except Exception:
                pass
            self._storage = None
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None

    def reset(self) -> None:
        """Drop the whole DB and start fresh."""
        self.close()
        self._scopes = {}
        self._open()

    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, *exc):  # pragma: no cover - trivial
        self.close()

    # ---- helpers --------------------------------------------------------

    def _scope_id(self, handle: str) -> str:
        sid = self._scopes.get(handle)
        if sid is None:
            raise KeyError(f"scope {handle!r} not registered — call ensure_scope first")
        return sid

    # ---- ReadOnlyAdapter -----------------------------------------------

    def health(self) -> HealthInfo:
        stats = self._storage.stats()
        chunks = self._storage.count_chunks()
        return HealthInfo(
            fragment_count=stats.get("fragments", 0),
            chunk_count=chunks,
            scope_count=stats.get("scopes", 0),
            version="ephemeral",
            tool=self.name,
            extra=dict(stats),
        )

    def recall(
        self,
        query: str,
        scope: str,
        *,
        limit: int = 10,
        types: list[str] | None = None,
    ) -> list[FragmentResult]:
        from skein.models import RecallRequest
        from skein.retrieval import recall as do_recall

        req = RecallRequest(query=query, scope=scope, types=types, limit=limit)
        response = do_recall(req, self._storage, self._provider)
        return [
            FragmentResult(
                id=r.fragment.id,
                content=r.fragment.content,
                type=r.fragment.type,
                score=float(r.score),
            )
            for r in response.results
        ]

    def search_code(
        self,
        query: str,
        scope: str,
        *,
        limit: int = 10,
    ) -> list[CodeChunkResult]:
        from skein.models import ChunkSearchRequest
        from skein.retrieval import search_chunks

        req = ChunkSearchRequest(query=query, scope=scope, limit=limit)
        response = search_chunks(req, self._storage, self._provider)
        return [
            CodeChunkResult(
                id=r.chunk.id,
                content=r.chunk.content,
                file_path=getattr(r.chunk, "file_path", ""),
                score=float(r.score),
            )
            for r in response.results
        ]

    # ---- MutableAdapter -------------------------------------------------

    def ensure_scope(self, handle: str, *, parent: str | None = None) -> str:
        from skein.models import ScopeCreate

        if handle in self._scopes:
            return self._scopes[handle]
        parent_id = self._scopes.get(parent) if parent else None
        scope_type = handle.split(":", 1)[0] if ":" in handle else "project"
        scope = self._storage.create_scope(ScopeCreate(
            handle=handle, type=scope_type, name=handle.split(":", 1)[-1],
            owner_id=self._user_id, parent_scope_id=parent_id,
        ))
        self._scopes[handle] = scope.id
        return scope.id

    def remember(
        self,
        content: str,
        *,
        type: str,
        scope: str,
        tags: list[str] | None = None,
        territory: str | None = None,
    ) -> str:
        from skein.models import FragmentCreate

        scope_id = self._scope_id(scope)
        frag = FragmentCreate(
            type=type, content=content, scope_id=scope_id, owner_id=self._user_id,
            tags=tags or [], territory=territory, extraction_method="explicit",
        )
        # Embed if provider produces real vectors; harmless if it doesn't.
        from skein.embeddings import vec_to_bytes
        embedding_bytes: bytes | None = None
        try:
            vec = self._provider.embed_one(content)
            embedding_bytes = vec_to_bytes(vec)
        except Exception:
            embedding_bytes = None
        stored = self._storage.create_fragment(frag, embedding=embedding_bytes)
        return stored.id

    def ingest_text(
        self,
        files: dict[str, str],
        *,
        scope: str,
        source_root: str = "bench",
    ) -> int:
        """Write the file map to a tmpdir and run the real ingest path."""
        from skein.ingest import ingest_directory

        scope_id = self._scope_id(scope)
        root = Path(tempfile.mkdtemp(prefix="bench_ingest_"))
        try:
            for rel, content in files.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            stats = ingest_directory(
                root, self._storage, self._provider,
                scope_id=scope_id, source_root=source_root,
            )
            return getattr(stats, "chunks_inserted", 0) + getattr(stats, "chunks_updated", 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # ---- Optional capabilities -----------------------------------------

    def would_capture_commit(self, subject: str, body: str = "") -> bool:
        from skein.git_watcher import GitCommit, is_noise_commit
        c = GitCommit(
            sha="0" * 40, author_name="bench", author_email="b@e",
            timestamp="2026-01-01T00:00:00Z", subject=subject, body=body,
        )
        return not is_noise_commit(c)

    def capture_git_commits(self, repo_path: str, *, scope: str) -> int:
        """Use the real watcher path. Returns how many commits became fragments."""
        from skein.git_watcher import (
            commit_to_fact,
            is_noise_commit,
            read_commits_since,
        )
        from skein.models import FragmentCreate

        commits = read_commits_since(Path(repo_path), limit=500)
        scope_id = self._scope_id(scope)
        created = 0
        for c in commits:
            if is_noise_commit(c):
                continue
            fact = commit_to_fact(c)
            frag = FragmentCreate(
                type="decision",
                content=fact.content,
                scope_id=scope_id,
                owner_id=self._user_id,
                tags=list(fact.tags),
                extraction_method="code-scanner",
                extraction_confidence=fact.confidence,
                created_against_commit=c.sha,
            )
            self._storage.create_fragment(frag)
            created += 1
        return created

    def claim_lease(self, glob: str, *, scope: str, ttl_seconds: int = 60) -> str | None:
        from skein.models import LeaseCreate

        scope_id = self._scope_id(scope)
        try:
            lease = self._storage.acquire_lease(LeaseCreate(
                scope_id=scope_id, glob=glob, owner_id=self._user_id,
                ttl_seconds=ttl_seconds,
            ))
            return lease.id
        except Exception:
            return None

    def release_lease(self, lease_id: str) -> None:
        try:
            self._storage.release_lease(lease_id, self._user_id)
        except Exception:
            pass
