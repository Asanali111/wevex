"""SQLite-backed storage layer for Skein.

This is the single source of truth for all persistence.  The retrieval module
builds on top for hybrid search.  The MCP server and REST routers call storage
methods — they never touch SQLite directly.

Thread safety: SQLite's WAL mode + check_same_thread=False is sufficient for
the single-process daemon with multiple async handlers.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Chunk, ChunkCreate,
    Commit, CommitCreate,
    Fragment, FragmentCreate, FragmentUpdate,
    Identity, IdentityCreate,
    Lease, LeaseCreate,
    Scope, ScopeCreate, ScopeMembership, ScopeMembershipCreate,
)

logger = logging.getLogger("skein.storage")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


class Storage:
    """Thread-safe SQLite storage for Skein."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,   # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._tune_pragmas()
        self._init_schema()

    # ------------------------------------------------------------------
    # PRAGMAs — applied per connection. WAL is set globally and persists,
    # but cache/mmap/busy_timeout are connection-scoped.
    # ------------------------------------------------------------------

    def _tune_pragmas(self) -> None:
        c = self._conn
        # Concurrent readers + a single writer (the daemon) co-existing with
        # the watcher subprocess. WAL is the right journal mode.
        c.execute("PRAGMA journal_mode=WAL")
        # Sufficient durability for our workload (lose at most last txn on
        # power loss). FULL is overkill for a local context store.
        c.execute("PRAGMA synchronous=NORMAL")
        # 30s grace before we surface "database is locked" — handles the
        # rare moment when the watcher and daemon both try to write.
        c.execute("PRAGMA busy_timeout=30000")
        # 64 MiB page cache — covers the entire working set for typical
        # projects (251 chunks ≈ a few MB).
        c.execute("PRAGMA cache_size=-65536")
        # 256 MiB memory-mapped read window — lets the kernel cache the DB
        # without an explicit read syscall on every page.
        c.execute("PRAGMA mmap_size=268435456")
        # FK enforcement is required for our schema's ON DELETE CASCADEs.
        c.execute("PRAGMA foreign_keys=ON")
        # Spill temp tables to memory rather than disk for the rare
        # CREATE TEMP TABLE (e.g. during VACUUM). Bounded by cache_size.
        c.execute("PRAGMA temp_store=MEMORY")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    # Module-level marker so the executescript only runs once per process per
    # DB file. The schema is `IF NOT EXISTS` everywhere so re-running is safe
    # but parsing the script costs ~5ms — adds up across hook invocations.
    _initialized_paths: set = set()

    def _init_schema(self) -> None:
        if self.db_path in Storage._initialized_paths:
            return
        schema_path = Path(__file__).parent / "schema.sql"
        sql = schema_path.read_text()
        # executescript handles multi-statement SQL (including triggers with
        # BEGIN…END blocks) correctly.  It also commits any open transaction.
        self._conn.executescript(sql)
        Storage._initialized_paths.add(self.db_path)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def create_identity(self, data: IdentityCreate) -> Identity:
        identity = Identity(**data.model_dump(), id=_new_id(), created_at=_now_iso())
        self._conn.execute(
            "INSERT INTO identities (id, handle, type, name, config, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (identity.id, identity.handle, identity.type, identity.name,
             json.dumps(identity.config), identity.created_at),
        )
        return identity

    def get_identity(self, id_or_handle: str) -> Optional[Identity]:
        row = self._conn.execute(
            "SELECT * FROM identities WHERE id = ? OR handle = ? LIMIT 1",
            (id_or_handle, id_or_handle),
        ).fetchone()
        return _row_to_identity(row) if row else None

    def get_or_create_identity(self, data: IdentityCreate) -> Identity:
        existing = self.get_identity(data.handle)
        if existing:
            return existing
        return self.create_identity(data)

    def list_identities(self, limit: int = 100, offset: int = 0) -> List[Identity]:
        rows = self._conn.execute(
            "SELECT * FROM identities ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_identity(r) for r in rows]

    def count_identities(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def create_scope(self, data: ScopeCreate) -> Scope:
        scope = Scope(**data.model_dump(), id=_new_id(), created_at=_now_iso())
        self._conn.execute(
            "INSERT INTO scopes (id, handle, type, name, parent_scope_id, owner_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (scope.id, scope.handle, scope.type, scope.name,
             scope.parent_scope_id, scope.owner_id, scope.created_at),
        )
        return scope

    def get_scope(self, id_or_handle: str) -> Optional[Scope]:
        row = self._conn.execute(
            "SELECT * FROM scopes WHERE id = ? OR handle = ? LIMIT 1",
            (id_or_handle, id_or_handle),
        ).fetchone()
        return _row_to_scope(row) if row else None

    def get_or_create_scope(self, data: ScopeCreate) -> Scope:
        existing = self.get_scope(data.handle)
        if existing:
            return existing
        return self.create_scope(data)

    def list_scopes(self, owner_id: Optional[str] = None,
                    limit: int = 100, offset: int = 0) -> List[Scope]:
        if owner_id:
            rows = self._conn.execute(
                "SELECT * FROM scopes WHERE owner_id = ? ORDER BY created_at DESC "
                "LIMIT ? OFFSET ?",
                (owner_id, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM scopes ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_scope(r) for r in rows]

    def count_scopes(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM scopes").fetchone()[0]

    def get_scope_lineage(self, handle: str) -> List[Scope]:
        """Return the scope and all its ancestors (self included), nearest first."""
        lineage: List[Scope] = []
        current = self.get_scope(handle)
        while current:
            lineage.append(current)
            if current.parent_scope_id:
                current = self.get_scope(current.parent_scope_id)
            else:
                break
        return lineage

    def add_scope_member(self, data: ScopeMembershipCreate) -> ScopeMembership:
        membership = ScopeMembership(**data.model_dump(), id=_new_id(), granted_at=_now_iso())
        self._conn.execute(
            "INSERT OR REPLACE INTO scope_memberships "
            "(id, scope_id, identity_id, role, granted_at) VALUES (?, ?, ?, ?, ?)",
            (membership.id, membership.scope_id, membership.identity_id,
             membership.role, membership.granted_at),
        )
        return membership

    # ------------------------------------------------------------------
    # Fragment (core write path)
    # ------------------------------------------------------------------

    def create_fragment(self, data: FragmentCreate, *,
                         commit_id: Optional[str] = None,
                         embedding: Optional[bytes] = None) -> Fragment:
        frag_id = _new_id()
        now = _now_iso()

        # Calculate expires_at if TTL is set
        expires_at: Optional[str] = None
        permanent = False
        if data.ttl_seconds is None:
            permanent = True
        elif data.ttl_seconds > 0:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=data.ttl_seconds)
            ).isoformat()

        self._conn.execute(
            """INSERT INTO fragments
               (id, type, content, scope_id, owner_id, confidence, version,
                ttl_seconds, expires_at, permanent, is_stale, stale_reason,
                tags, territory, source_commit_id, metadata, content_embedding,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 0, NULL,
                       ?, ?, ?, ?, ?, ?, ?)""",
            (frag_id, data.type, data.content, data.scope_id, data.owner_id,
             data.confidence, data.ttl_seconds, expires_at,
             1 if permanent else 0,
             json.dumps(data.tags), data.territory, commit_id,
             json.dumps(data.metadata), embedding,
             now, now),
        )

        return Fragment(
            id=frag_id, type=data.type, content=data.content,
            scope_id=data.scope_id, owner_id=data.owner_id,
            confidence=data.confidence, version=1,
            ttl_seconds=data.ttl_seconds, expires_at=expires_at,
            permanent=permanent, is_stale=False,
            tags=data.tags, territory=data.territory,
            source_commit_id=commit_id, metadata=data.metadata,
            created_at=now, updated_at=now,
        )

    def update_fragment(self, frag_id: str, data: FragmentUpdate) -> Fragment:
        """PATCH-style update with OCC version check.

        Uses an atomic ``UPDATE … WHERE id = ? AND version = ?`` so the version
        check and the write happen in a single statement. Two concurrent writers
        starting from the same version will see exactly one win — the loser's
        UPDATE matches zero rows and we raise ``ConflictError``.
        """
        row = self._conn.execute(
            "SELECT * FROM fragments WHERE id = ?", (frag_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Fragment {frag_id!r} not found")

        current_version = row["version"]
        if current_version != data.expected_version:
            raise ConflictError(
                f"OCC conflict: expected version {data.expected_version}, "
                f"current is {current_version}",
                code="VERSION_CONFLICT",
            )

        updates: Dict[str, Any] = {}
        if data.content is not None:
            updates["content"] = data.content
        if data.confidence is not None:
            updates["confidence"] = data.confidence
        if data.tags is not None:
            updates["tags"] = json.dumps(data.tags)
        if data.territory is not None:
            updates["territory"] = data.territory
        if data.is_stale is not None:
            updates["is_stale"] = 1 if data.is_stale else 0
        if data.stale_reason is not None:
            updates["stale_reason"] = data.stale_reason
        if data.metadata is not None:
            updates["metadata"] = json.dumps(data.metadata)

        updates["version"] = current_version + 1
        updates["updated_at"] = _now_iso()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [frag_id, current_version]
        n = self._conn.execute(
            f"UPDATE fragments SET {set_clause} "
            f"WHERE id = ? AND version = ?",
            values,
        ).rowcount
        if n == 0:
            # Another writer beat us between the SELECT and the UPDATE.
            raise ConflictError(
                "OCC conflict: row was modified by a concurrent writer",
                code="VERSION_CONFLICT",
            )

        return self.get_fragment(frag_id)  # type: ignore[return-value]

    def get_fragment(self, frag_id: str) -> Optional[Fragment]:
        row = self._conn.execute(
            "SELECT * FROM fragments WHERE id = ?", (frag_id,)
        ).fetchone()
        return _row_to_fragment(row) if row else None

    def delete_fragment(self, frag_id: str) -> bool:
        """Soft-delete: mark as stale.  Hard-delete is manual/admin only."""
        n = self._conn.execute(
            "UPDATE fragments SET is_stale = 1, stale_reason = 'deleted', "
            "updated_at = ? WHERE id = ? AND is_stale = 0",
            (_now_iso(), frag_id),
        ).rowcount
        return n > 0

    def list_fragments(self, scope_id: Optional[str] = None,
                        type_filter: Optional[str] = None,
                        include_stale: bool = False,
                        limit: int = 50,
                        offset: int = 0) -> List[Fragment]:
        conditions = ["1=1"]
        params: List[Any] = []

        if scope_id:
            conditions.append("scope_id = ?")
            params.append(scope_id)
        if type_filter:
            conditions.append("type = ?")
            params.append(type_filter)
        if not include_stale:
            conditions.append("is_stale = 0")
            conditions.append(
                "(expires_at IS NULL OR expires_at > datetime('now'))"
            )

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM fragments WHERE {where} ORDER BY created_at DESC "
            f"LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [_row_to_fragment(r) for r in rows]

    def count_fragments(self, include_stale: bool = False) -> int:
        if include_stale:
            return self._conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM fragments WHERE is_stale = 0 "
            "AND (expires_at IS NULL OR expires_at > datetime('now'))"
        ).fetchone()[0]

    def count_fragments_in_scope(self, scope_id: str, include_stale: bool = False) -> int:
        """Cheap existence check used by hot-path hooks to early-exit when the
        scope has no fragments — avoids the embedding+BM25+vector roundtrip."""
        if include_stale:
            return self._conn.execute(
                "SELECT COUNT(*) FROM fragments WHERE scope_id = ?", (scope_id,),
            ).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM fragments WHERE scope_id = ? "
            "AND is_stale = 0 "
            "AND (expires_at IS NULL OR expires_at > datetime('now'))",
            (scope_id,),
        ).fetchone()[0]

    def get_fragment_embedding(self, frag_id: str) -> Optional[bytes]:
        row = self._conn.execute(
            "SELECT content_embedding FROM fragments WHERE id = ?", (frag_id,)
        ).fetchone()
        if row is None:
            return None
        return row[0]

    def set_fragment_embedding(self, frag_id: str, embedding_bytes: bytes) -> None:
        self._conn.execute(
            "UPDATE fragments SET content_embedding = ? WHERE id = ?",
            (embedding_bytes, frag_id),
        )

    def get_fragments_without_embeddings(self, limit: int = 100) -> List[Fragment]:
        rows = self._conn.execute(
            "SELECT * FROM fragments WHERE content_embedding IS NULL "
            "AND is_stale = 0 ORDER BY created_at LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_fragment(r) for r in rows]

    # ------------------------------------------------------------------
    # Keyword search (FTS5 BM25)
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, scope_ids: List[str],
                        type_filter: Optional[List[str]] = None,
                        include_stale: bool = False,
                        limit: int = 20) -> List[Tuple[str, float]]:
        """Return (fragment_id, bm25_score) pairs ordered by relevance.

        BM25 score from FTS5 is negative (more negative = more relevant).
        We negate it so higher = better.
        """
        if not scope_ids:
            return []

        placeholders = ",".join("?" * len(scope_ids))
        type_filter_clause = ""
        type_params: List[Any] = []
        if type_filter:
            tp = ",".join("?" * len(type_filter))
            type_filter_clause = f"AND f.type IN ({tp})"
            type_params = list(type_filter)

        stale_clause = "" if include_stale else (
            "AND f.is_stale = 0 "
            "AND (f.expires_at IS NULL OR f.expires_at > datetime('now'))"
        )

        rows = self._conn.execute(
            f"""
            SELECT fts.fragment_id, -bm25(fragments_fts) AS score
            FROM fragments_fts AS fts
            JOIN fragments AS f ON f.id = fts.fragment_id
            WHERE fragments_fts MATCH ?
              AND f.scope_id IN ({placeholders})
              {type_filter_clause}
              {stale_clause}
            ORDER BY bm25(fragments_fts)
            LIMIT ?
            """,
            [_fts_escape(query)] + scope_ids + type_params + [limit],
        ).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    # ------------------------------------------------------------------
    # Vector search (cosine, Python-side)
    # ------------------------------------------------------------------

    def vector_search(self, query_vec_bytes: bytes, scope_ids: List[str],
                       type_filter: Optional[List[str]] = None,
                       include_stale: bool = False,
                       limit: int = 20,
                       dimension: int = 768) -> List[Tuple[str, float]]:
        """Return (fragment_id, cosine_similarity) ordered by score.

        Loads all embeddings for the scope into memory and computes cosine
        similarity Python-side.  This is fine at <50k fragments; add ANN
        indexing (e.g. usearch or sqlite-vec) for larger datasets.
        """
        import numpy as np
        from .embeddings import bytes_to_vec, cosine_similarity

        if not scope_ids:
            return []

        placeholders = ",".join("?" * len(scope_ids))
        type_filter_clause = ""
        type_params: List[Any] = []
        if type_filter:
            tp = ",".join("?" * len(type_filter))
            type_filter_clause = f"AND type IN ({tp})"
            type_params = list(type_filter)

        stale_clause = "" if include_stale else (
            "AND is_stale = 0 "
            "AND (expires_at IS NULL OR expires_at > datetime('now'))"
        )

        rows = self._conn.execute(
            f"""SELECT id, content_embedding FROM fragments
                WHERE content_embedding IS NOT NULL
                  AND scope_id IN ({placeholders})
                  {type_filter_clause}
                  {stale_clause}""",
            scope_ids + type_params,
        ).fetchall()

        if not rows:
            return []

        query_vec = bytes_to_vec(query_vec_bytes, dimension)
        scored: List[Tuple[str, float]] = []
        for row in rows:
            frag_vec = bytes_to_vec(row[1], dimension)
            sim = cosine_similarity(query_vec, frag_vec)
            scored.append((row[0], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def create_commit(self, data: CommitCreate) -> Commit:
        commit = Commit(**data.model_dump(), id=_new_id(), created_at=_now_iso())
        self._conn.execute(
            """INSERT INTO commits
               (id, author_id, scope_id, parent_commit_id, message,
                fragments_added, fragments_modified, fragments_removed,
                metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (commit.id, commit.author_id, commit.scope_id,
             commit.parent_commit_id, commit.message,
             json.dumps(commit.fragments_added),
             json.dumps(commit.fragments_modified),
             json.dumps(commit.fragments_removed),
             json.dumps(commit.metadata), commit.created_at),
        )
        return commit

    def get_commit(self, commit_id: str) -> Optional[Commit]:
        row = self._conn.execute(
            "SELECT * FROM commits WHERE id = ?", (commit_id,)
        ).fetchone()
        return _row_to_commit(row) if row else None

    def list_commits(self, scope_id: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> List[Commit]:
        if scope_id:
            rows = self._conn.execute(
                "SELECT * FROM commits WHERE scope_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (scope_id, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM commits ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_commit(r) for r in rows]

    # ------------------------------------------------------------------
    # Lease
    # ------------------------------------------------------------------

    def acquire_lease(self, data: LeaseCreate) -> Lease:
        lease_id = _new_id()
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=data.ttl_seconds)).isoformat()
        now_iso = now.isoformat()

        self._conn.execute(
            """INSERT INTO leases
               (id, scope_id, glob, owner_id, reason, acquired_at, expires_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (lease_id, data.scope_id, data.glob, data.owner_id,
             data.reason, now_iso, expires_at, json.dumps(data.metadata)),
        )

        return Lease(
            id=lease_id, scope_id=data.scope_id, glob=data.glob,
            owner_id=data.owner_id, ttl_seconds=data.ttl_seconds,
            reason=data.reason, metadata=data.metadata,
            acquired_at=now_iso, expires_at=expires_at,
        )

    def get_lease(self, lease_id: str) -> Optional[Lease]:
        row = self._conn.execute(
            "SELECT * FROM leases WHERE id = ?", (lease_id,)
        ).fetchone()
        return _row_to_lease(row) if row else None

    def release_lease(self, lease_id: str, owner_id: str) -> bool:
        n = self._conn.execute(
            "DELETE FROM leases WHERE id = ? AND owner_id = ?",
            (lease_id, owner_id),
        ).rowcount
        return n > 0

    def list_leases(self, scope_id: Optional[str] = None,
                     active_only: bool = True) -> List[Lease]:
        conditions = ["1=1"]
        params: List[Any] = []
        if scope_id:
            conditions.append("scope_id = ?")
            params.append(scope_id)
        if active_only:
            conditions.append("expires_at > datetime('now')")

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM leases WHERE {where} ORDER BY acquired_at DESC",
            params,
        ).fetchall()
        return [_row_to_lease(r) for r in rows]

    def cleanup_expired_leases(self) -> int:
        n = self._conn.execute(
            "DELETE FROM leases WHERE expires_at <= datetime('now')"
        ).rowcount
        if n:
            logger.debug("Cleaned up %d expired leases", n)
        return n

    def check_lease_conflict(self, scope_id: str, glob: str) -> Optional[Lease]:
        """Return an active conflicting lease (glob overlap) if one exists."""
        # Simple substring match — for exact overlaps.
        # Full glob-vs-glob overlap detection would require fnmatch iteration.
        rows = self._conn.execute(
            "SELECT * FROM leases WHERE scope_id = ? AND expires_at > datetime('now')",
            (scope_id,),
        ).fetchall()
        import fnmatch
        for row in rows:
            other = _row_to_lease(row)
            # Check if either glob matches the other as a path prefix/pattern
            if fnmatch.fnmatch(glob, other.glob) or fnmatch.fnmatch(other.glob, glob):
                return other
            # Also check if they share a common prefix path component
            if _glob_overlaps(glob, other.glob):
                return other
        return None

    # ------------------------------------------------------------------
    # Chunks (codebase RAG)
    # ------------------------------------------------------------------

    def upsert_chunk(self, data: ChunkCreate, *,
                     content_hash: str,
                     embedding: Optional[bytes] = None) -> tuple[Chunk, str]:
        """Insert/update a chunk keyed by (scope, root, path, line range).

        Returns ``(chunk, status)`` where ``status`` is one of:
          - ``"inserted"`` — no row existed at this location
          - ``"updated"``  — existed but content_hash differed
          - ``"unchanged"`` — existed with the same content_hash (no-op)
        """
        existing = self._conn.execute(
            "SELECT * FROM chunks WHERE scope_id = ? AND source_root = ? "
            "AND source_path = ? AND line_start = ? AND line_end = ?",
            (data.scope_id, data.source_root, data.source_path,
             data.line_start, data.line_end),
        ).fetchone()

        if existing and existing["content_hash"] == content_hash:
            return _row_to_chunk(existing), "unchanged"

        if existing:
            self._conn.execute(
                """UPDATE chunks SET
                       content = ?, content_hash = ?, language = ?,
                       chunk_type = ?, symbol_name = ?, metadata = ?,
                       content_embedding = ?
                   WHERE id = ?""",
                (data.content, content_hash, data.language,
                 data.chunk_type, data.symbol_name,
                 json.dumps(data.metadata), embedding, existing["id"]),
            )
            return self._get_chunk_by_id(existing["id"]), "updated"

        chunk_id = _new_id()
        now = _now_iso()
        self._conn.execute(
            """INSERT INTO chunks
               (id, scope_id, source_root, source_path, language, chunk_type,
                symbol_name, line_start, line_end, content, content_hash,
                content_embedding, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, data.scope_id, data.source_root, data.source_path,
             data.language, data.chunk_type, data.symbol_name,
             data.line_start, data.line_end, data.content, content_hash,
             embedding, json.dumps(data.metadata), now),
        )
        return Chunk(
            id=chunk_id,
            scope_id=data.scope_id, source_root=data.source_root,
            source_path=data.source_path, language=data.language,
            chunk_type=data.chunk_type, symbol_name=data.symbol_name,
            line_start=data.line_start, line_end=data.line_end,
            content=data.content, content_hash=content_hash,
            metadata=data.metadata, created_at=now,
        ), "inserted"

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._get_chunk_by_id(chunk_id)

    def _get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def list_chunks(self, scope_id: Optional[str] = None,
                     source_root: Optional[str] = None,
                     language: Optional[str] = None,
                     limit: int = 50, offset: int = 0) -> List[Chunk]:
        conditions, params = ["1=1"], []
        if scope_id:
            conditions.append("scope_id = ?")
            params.append(scope_id)
        if source_root:
            conditions.append("source_root = ?")
            params.append(source_root)
        if language:
            conditions.append("language = ?")
            params.append(language)
        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE {where} "
            f"ORDER BY source_path, line_start LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def delete_chunks_by_root(self, scope_id: str, source_root: str) -> int:
        n = self._conn.execute(
            "DELETE FROM chunks WHERE scope_id = ? AND source_root = ?",
            (scope_id, source_root),
        ).rowcount
        return n

    def chunk_stats(self, scope_id: Optional[str] = None) -> Dict[str, Any]:
        scope_clause = ""
        params: List[Any] = []
        if scope_id:
            scope_clause = "WHERE scope_id = ?"
            params = [scope_id]

        total = self._conn.execute(
            f"SELECT COUNT(*) FROM chunks {scope_clause}", params,
        ).fetchone()[0]
        files = self._conn.execute(
            f"SELECT COUNT(DISTINCT source_path) FROM chunks {scope_clause}", params,
        ).fetchone()[0]
        by_lang = dict(self._conn.execute(
            f"SELECT COALESCE(language, 'unknown'), COUNT(*) FROM chunks "
            f"{scope_clause} GROUP BY language",
            params,
        ).fetchall())
        by_root = dict(self._conn.execute(
            f"SELECT source_root, COUNT(*) FROM chunks {scope_clause} "
            f"GROUP BY source_root",
            params,
        ).fetchall())
        return {
            "total_chunks": total,
            "total_files": files,
            "by_language": by_lang,
            "by_root": by_root,
        }

    def get_chunks_by_ids(self, ids: List[str]) -> Dict[str, Chunk]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids,
        ).fetchall()
        return {r["id"]: _row_to_chunk(r) for r in rows}

    def bulk_get_chunk_hashes(
        self,
        scope_id: str,
        source_root: str,
        keys: List[Tuple[str, int, int]],
    ) -> Dict[Tuple[str, int, int], str]:
        """Bulk lookup of existing chunk content_hashes.

        Used by ingest to decide which chunks in a batch are unchanged
        (hash matches) and can therefore skip both embedding and the upsert
        round-trip. Single set-based query so a batch of 32 chunks costs
        one SELECT instead of 32. Returns ``{(path, line_start, line_end):
        hash}``."""
        if not keys:
            return {}
        or_clauses = " OR ".join(
            ["(source_path = ? AND line_start = ? AND line_end = ?)"] * len(keys)
        )
        flat: List[Any] = [scope_id, source_root]
        for k in keys:
            flat.extend(k)
        rows = self._conn.execute(
            f"""SELECT source_path, line_start, line_end, content_hash
                FROM chunks
                WHERE scope_id = ? AND source_root = ? AND ({or_clauses})""",
            flat,
        ).fetchall()
        return {(r[0], r[1], r[2]): r[3] for r in rows}

    def begin_immediate(self) -> None:
        """Begin an explicit transaction for batched writes. Pair with
        ``commit_immediate()`` / ``rollback_immediate()``. Lets callers
        amortise the per-statement fsync cost across a batch.

        Routed through Storage so test fakes can no-op it."""
        self._conn.execute("BEGIN IMMEDIATE")

    def commit_immediate(self) -> None:
        self._conn.execute("COMMIT")

    def rollback_immediate(self) -> None:
        self._conn.execute("ROLLBACK")

    def chunks_keyword_search(
        self, query: str, scope_ids: List[str], *,
        languages: Optional[List[str]] = None,
        source_root: Optional[str] = None,
        limit: int = 30,
    ) -> List[Tuple[str, float]]:
        if not scope_ids:
            return []
        scope_placeholders = ",".join("?" * len(scope_ids))
        lang_clause, lang_params = "", []
        if languages:
            lp = ",".join("?" * len(languages))
            lang_clause = f"AND c.language IN ({lp})"
            lang_params = list(languages)
        root_clause, root_params = "", []
        if source_root:
            root_clause = "AND c.source_root = ?"
            root_params = [source_root]
        rows = self._conn.execute(
            f"""SELECT fts.chunk_id, -bm25(chunks_fts) AS score
                FROM chunks_fts AS fts
                JOIN chunks AS c ON c.id = fts.chunk_id
                WHERE chunks_fts MATCH ?
                  AND c.scope_id IN ({scope_placeholders})
                  {lang_clause}
                  {root_clause}
                ORDER BY bm25(chunks_fts)
                LIMIT ?""",
            [_fts_escape(query)] + scope_ids + lang_params + root_params + [limit],
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def chunks_vector_search(
        self, query_vec_bytes: bytes, scope_ids: List[str], *,
        languages: Optional[List[str]] = None,
        source_root: Optional[str] = None,
        limit: int = 30,
        dimension: int = 768,
        batch_size: int = 5000,
    ) -> List[Tuple[str, float]]:
        """Brute-force cosine search, batched to bound memory.

        Streams rows in ``batch_size`` chunks and keeps a heap of top ``limit``.
        Memory: O(batch_size · dimension · 4 bytes). For 5000 chunks at 768
        dims that's ~15 MB per batch — fine on any laptop.
        """
        import heapq
        import numpy as np
        from .embeddings import bytes_to_vec

        if not scope_ids:
            return []

        scope_placeholders = ",".join("?" * len(scope_ids))
        lang_clause, lang_params = "", []
        if languages:
            lp = ",".join("?" * len(languages))
            lang_clause = f"AND language IN ({lp})"
            lang_params = list(languages)
        root_clause, root_params = "", []
        if source_root:
            root_clause = "AND source_root = ?"
            root_params = [source_root]

        # Normalised query vector for cosine via dot product
        query_vec = bytes_to_vec(query_vec_bytes, dimension)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        q_unit = query_vec / q_norm

        cur = self._conn.execute(
            f"""SELECT id, content_embedding FROM chunks
                WHERE content_embedding IS NOT NULL
                  AND scope_id IN ({scope_placeholders})
                  {lang_clause}
                  {root_clause}""",
            scope_ids + lang_params + root_params,
        )

        # min-heap of size <= limit: (score, id)
        heap: List[Tuple[float, str]] = []
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            ids = [r[0] for r in rows]
            mat = np.frombuffer(
                b"".join(r[1] for r in rows), dtype=np.float32,
            ).reshape(-1, dimension)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = mat @ q_unit / norms
            for cid, s in zip(ids, sims.tolist()):
                if len(heap) < limit:
                    heapq.heappush(heap, (s, cid))
                else:
                    if s > heap[0][0]:
                        heapq.heapreplace(heap, (s, cid))

        return sorted(((cid, float(s)) for s, cid in heap), key=lambda x: -x[1])

    def count_chunks(self, scope_id: Optional[str] = None) -> int:
        if scope_id:
            return self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE scope_id = ?", (scope_id,),
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ------------------------------------------------------------------
    # Mark stale (scheduled maintenance)
    # ------------------------------------------------------------------

    def mark_expired_fragments_stale(self) -> int:
        n = self._conn.execute(
            "UPDATE fragments SET is_stale = 1, stale_reason = 'ttl_expired' "
            "WHERE is_stale = 0 AND expires_at IS NOT NULL "
            "AND expires_at <= datetime('now')",
        ).rowcount
        if n:
            logger.info("Marked %d fragments stale (TTL expired)", n)
        return n

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "fragments": self.count_fragments(),
            "fragments_stale": self._conn.execute(
                "SELECT COUNT(*) FROM fragments WHERE is_stale = 1"
            ).fetchone()[0],
            "scopes": self.count_scopes(),
            "identities": self.count_identities(),
            "commits": self._conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0],
            "leases": self._conn.execute(
                "SELECT COUNT(*) FROM leases WHERE expires_at > datetime('now')"
            ).fetchone()[0],
        }

    # ------------------------------------------------------------------
    # Raw access (for retrieval module)
    # ------------------------------------------------------------------

    def get_fragments_by_ids(self, ids: List[str]) -> Dict[str, Fragment]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM fragments WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {row["id"]: _row_to_fragment(row) for row in rows}

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ConflictError(Exception):
    def __init__(self, message: str, code: str = "CONFLICT") -> None:
        super().__init__(message)
        self.code = code


class NotFoundError(Exception):
    pass


# ---------------------------------------------------------------------------
# Row → model helpers
# ---------------------------------------------------------------------------

def _row_to_identity(row: sqlite3.Row) -> Identity:
    return Identity(
        id=row["id"], handle=row["handle"], type=row["type"], name=row["name"],
        config=json.loads(row["config"]), created_at=row["created_at"],
    )


def _row_to_scope(row: sqlite3.Row) -> Scope:
    return Scope(
        id=row["id"], handle=row["handle"], type=row["type"], name=row["name"],
        parent_scope_id=row["parent_scope_id"], owner_id=row["owner_id"],
        created_at=row["created_at"],
    )


def _row_to_commit(row: sqlite3.Row) -> Commit:
    return Commit(
        id=row["id"], author_id=row["author_id"], scope_id=row["scope_id"],
        parent_commit_id=row["parent_commit_id"], message=row["message"],
        fragments_added=json.loads(row["fragments_added"]),
        fragments_modified=json.loads(row["fragments_modified"]),
        fragments_removed=json.loads(row["fragments_removed"]),
        metadata=json.loads(row["metadata"]), created_at=row["created_at"],
    )


def _row_to_fragment(row: sqlite3.Row) -> Fragment:
    return Fragment(
        id=row["id"], type=row["type"], content=row["content"],
        scope_id=row["scope_id"], owner_id=row["owner_id"],
        confidence=row["confidence"], version=row["version"],
        ttl_seconds=row["ttl_seconds"], expires_at=row["expires_at"],
        permanent=bool(row["permanent"]), is_stale=bool(row["is_stale"]),
        stale_reason=row["stale_reason"],
        tags=json.loads(row["tags"]),
        territory=row["territory"],
        source_commit_id=row["source_commit_id"],
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _row_to_lease(row: sqlite3.Row) -> Lease:
    return Lease(
        id=row["id"], scope_id=row["scope_id"], glob=row["glob"],
        owner_id=row["owner_id"], reason=row["reason"],
        ttl_seconds=0,   # we store expires_at, not the original TTL
        metadata=json.loads(row["metadata"]),
        acquired_at=row["acquired_at"], expires_at=row["expires_at"],
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"], scope_id=row["scope_id"],
        source_root=row["source_root"], source_path=row["source_path"],
        language=row["language"], chunk_type=row["chunk_type"],
        symbol_name=row["symbol_name"],
        line_start=row["line_start"], line_end=row["line_end"],
        content=row["content"], content_hash=row["content_hash"],
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# SQL utilities
# ---------------------------------------------------------------------------

def _fts_escape(query: str) -> str:
    """Escape special FTS5 characters and append wildcard for prefix match."""
    # Remove special FTS5 chars that would cause parse errors
    for ch in ('"', "'", "*", "^", ":", "(", ")"):
        query = query.replace(ch, " ")
    parts = query.split()
    if not parts:
        return '""'
    # Wrap each token in quotes and add trailing wildcard for prefix search
    return " ".join(f'"{p}"*' for p in parts if p)


def _glob_overlaps(a: str, b: str) -> bool:
    """Heuristic: do two glob patterns share a path prefix (tree overlap)?

    Compares path *segments*, not raw strings — so ``"a/bc/**"`` does NOT
    falsely overlap ``"a/b/**"`` the way naive ``startswith`` did.
    """
    def _segments(g: str) -> List[str]:
        # Strip any trailing wildcards and slashes, then split into segments.
        base = g
        while base and base[-1] in "/*":
            base = base[:-1]
        return [p for p in base.split("/") if p]

    a_parts = _segments(a)
    b_parts = _segments(b)
    if not a_parts or not b_parts:
        return False
    shorter, longer = (a_parts, b_parts) if len(a_parts) <= len(b_parts) else (b_parts, a_parts)
    return longer[:len(shorter)] == shorter
