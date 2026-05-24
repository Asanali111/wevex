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
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

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
        # Iter 31: per-instance id namespaces the retrieval recall cache so
        # ephemeral test adapters and the daemon's primary Storage don't
        # share keys. Without this, a `recall(query, scope)` cached against
        # one Storage instance would be returned for the same query
        # against a completely different Storage on the same scope handle.
        self.instance_id = uuid.uuid4().hex
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
        # Iter 31: page cache trimmed 64 MiB → 16 MiB. The single-user
        # daemon's working set fits in a few MB; the extra 48 MiB was
        # giving "lightweight" the lie. Negative sign means KiB. Tunable
        # via SKEIN_SQLITE_CACHE_KB for power users with huge stores.
        cache_kib = int(os.environ.get("SKEIN_SQLITE_CACHE_KB", "16384"))
        c.execute(f"PRAGMA cache_size=-{cache_kib}")
        # Iter 31: mmap window 256 MiB → 64 MiB. Same reasoning —
        # 256 MiB was sized for a hypothetical multi-gig store; we're
        # at ~40 MiB today. Tunable via SKEIN_SQLITE_MMAP_MB.
        mmap_mb = int(os.environ.get("SKEIN_SQLITE_MMAP_MB", "64"))
        c.execute(f"PRAGMA mmap_size={mmap_mb * 1024 * 1024}")
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
        # Pre-migrations FIRST — schema.sql references the provenance columns
        # in CREATE INDEX statements, which would fail on a legacy DB that
        # hasn't had them added yet. So we add the columns first, then run
        # the IF-NOT-EXISTS schema script second.
        if self._fragments_table_exists():
            self._migrate_provenance_columns()
        schema_path = Path(__file__).parent / "schema.sql"
        sql = schema_path.read_text()
        # executescript handles multi-statement SQL (including triggers with
        # BEGIN…END blocks) correctly.  It also commits any open transaction.
        self._conn.executescript(sql)
        Storage._initialized_paths.add(self.db_path)

    def _fragments_table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fragments'"
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # MCP client tracking (iter 14.0b)
    # Maps a bearer token prefix → which LLM client introduced itself in
    # its ``initialize`` handshake. Lets us tag every fragment with its
    # originating tool (``claude-code``, ``cursor``, …) without making the
    # user manage per-client tokens.
    # ------------------------------------------------------------------

    def upsert_mcp_client(self, token_prefix: str, client_name: str,
                          display_name: Optional[str] = None) -> None:
        """Register or update the client for this token prefix."""
        self._conn.execute(
            """INSERT INTO mcp_clients (token_prefix, client_name, display_name, created_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(token_prefix) DO UPDATE SET
                 client_name = excluded.client_name,
                 display_name = COALESCE(excluded.display_name, mcp_clients.display_name)""",
            (token_prefix, client_name, display_name),
        )

    def get_client_for_token_prefix(self, token_prefix: str) -> Optional[str]:
        """Return the registered client_name for this token prefix, or None."""
        row = self._conn.execute(
            "SELECT client_name FROM mcp_clients WHERE token_prefix = ?",
            (token_prefix,),
        ).fetchone()
        return row["client_name"] if row else None

    def list_mcp_clients(self) -> list[dict[str, Any]]:
        """All registered clients, newest first. Used by `skein clients`."""
        rows = self._conn.execute(
            "SELECT token_prefix, client_name, display_name, created_at "
            "FROM mcp_clients ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Extraction candidates (iter 14.1 / 14.2)
    # The inbox queue: passive watchers (code scanner, transcript extractor)
    # write here. Users approve via ``skein inbox approve <id>``.
    # ------------------------------------------------------------------

    def add_extraction_candidate(
        self, *, scope_id: str, content: str, type: str,
        confidence: float, source_tool: str,
        territory: Optional[str] = None, tags: Optional[list[str]] = None,
        source_session_id: Optional[str] = None,
        source_file: Optional[str] = None,
        source_message_ts: Optional[str] = None,
    ) -> Optional[str]:
        """Insert a pending candidate. Returns the new id, or None if a
        duplicate already exists (dedup is by scope_id + content + source_tool).
        """
        candidate_id = _new_id()
        try:
            self._conn.execute(
                """INSERT INTO extraction_candidates
                   (id, scope_id, content, type, territory, tags, confidence,
                    source_tool, source_session_id, source_file, source_message_ts,
                    status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', datetime('now'))""",
                (candidate_id, scope_id, content, type, territory,
                 json.dumps(tags or []), confidence,
                 source_tool, source_session_id, source_file, source_message_ts),
            )
            return candidate_id
        except sqlite3.IntegrityError:
            # Dedup hit: same (scope, content, tool) already queued — skip silently
            return None

    def list_extraction_candidates(
        self, *, status: str = "pending", scope_id: Optional[str] = None,
        limit: int = 50, offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions = ["status = ?"]
        params: list[Any] = [status]
        if scope_id:
            conditions.append("scope_id = ?")
            params.append(scope_id)
        where = " AND ".join(conditions)
        params.extend([limit, offset])
        rows = self._conn.execute(
            f"SELECT * FROM extraction_candidates WHERE {where} "
            f"ORDER BY confidence DESC, created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_extraction_candidate(self, candidate_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM extraction_candidates WHERE id = ?", (candidate_id,),
        ).fetchone()
        return dict(row) if row else None

    def mark_candidate_status(
        self, candidate_id: str, status: str,
        promoted_fragment_id: Optional[str] = None,
    ) -> bool:
        n = self._conn.execute(
            "UPDATE extraction_candidates SET status = ?, reviewed_at = datetime('now'), "
            "promoted_fragment_id = COALESCE(?, promoted_fragment_id) "
            "WHERE id = ? AND status = 'pending'",
            (status, promoted_fragment_id, candidate_id),
        ).rowcount
        return n > 0

    def count_extraction_candidates(self, status: str = "pending") -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM extraction_candidates WHERE status = ?",
            (status,),
        ).fetchone()
        return int(row["c"])

    # ------------------------------------------------------------------
    # Transcript cursors (iter 14.2)
    # ------------------------------------------------------------------

    def get_transcript_cursor(self, file_path: str) -> int:
        row = self._conn.execute(
            "SELECT last_byte_offset FROM transcript_cursors WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return int(row["last_byte_offset"]) if row else 0

    def set_transcript_cursor(self, file_path: str, last_byte_offset: int,
                               client_name: str = "claude-code") -> None:
        self._conn.execute(
            """INSERT INTO transcript_cursors (file_path, last_byte_offset, last_seen_at, client_name)
               VALUES (?, ?, datetime('now'), ?)
               ON CONFLICT(file_path) DO UPDATE SET
                 last_byte_offset = excluded.last_byte_offset,
                 last_seen_at = datetime('now')""",
            (file_path, last_byte_offset, client_name),
        )

    # ------------------------------------------------------------------
    # AGENTS.md render state (iter 26 / ADR-002)
    # Daemon auto-sync uses this to skip regen when nothing changed.
    # ------------------------------------------------------------------

    def get_agents_md_state(
        self, scope_handle: str, file_path: str,
    ) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM agents_md_state WHERE scope_handle = ? AND file_path = ?",
            (scope_handle, file_path),
        ).fetchone()
        return dict(row) if row else None

    def upsert_agents_md_state(
        self, scope_handle: str, file_path: str, render_hash: str,
    ) -> None:
        self._conn.execute(
            """INSERT INTO agents_md_state (scope_handle, file_path, last_render_hash, last_render_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(scope_handle, file_path) DO UPDATE SET
                 last_render_hash = excluded.last_render_hash,
                 last_render_at = datetime('now')""",
            (scope_handle, file_path, render_hash),
        )

    # ------------------------------------------------------------------
    # Migrations — kept inline because Skein deliberately ships a single
    # SQLite file and doesn't want an Alembic-style migrations dir.
    # Each migration helper is a no-op if it's already been applied.
    # ------------------------------------------------------------------

    def _migrate_provenance_columns(self) -> None:
        """Add iter-14.0 provenance columns + iter-25 ``value`` to ``fragments``
        if missing.

        New installs get the columns via ``schema.sql``'s CREATE TABLE; old
        installs (anything created before the column existed) need ALTER
        TABLE. SQLite ALTER TABLE doesn't support IF NOT EXISTS, so we
        introspect first. When ``value`` is added to an existing DB, we
        backfill every live row from its provenance + type + content so the
        column isn't uniformly 0.5 the day it ships.
        """
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(fragments)")
        }
        wanted: list[tuple] = [
            ("created_by_tool",          "TEXT"),
            ("created_in_session_id",    "TEXT"),
            ("created_against_commit",   "TEXT"),
            ("files_open_at_creation",   "TEXT NOT NULL DEFAULT '[]'"),
            ("supersedes_fragment_id",   "TEXT REFERENCES fragments(id)"),
            ("superseded_by_fragment_id","TEXT REFERENCES fragments(id)"),
            ("extraction_method",        "TEXT NOT NULL DEFAULT 'explicit'"),
            ("extraction_confidence",    "REAL"),
            ("value",                    "REAL NOT NULL DEFAULT 0.5"),
            # Iter 31 — efficiency pass. Three new columns: dedupe_key
            # lets create_fragment short-circuit identical writes (no
            # duplicate rows when remember() is called twice with the same
            # content); recall_hits + last_recalled_at feed the
            # behavioural-value loop so fragments that actually get used
            # rise to the top organically.
            ("dedupe_key",               "TEXT"),
            ("recall_hits",              "INTEGER NOT NULL DEFAULT 0"),
            ("last_recalled_at",         "TEXT"),
        ]
        value_just_added = False
        for col_name, col_def in wanted:
            if col_name in existing:
                continue
            try:
                self._conn.execute(
                    f"ALTER TABLE fragments ADD COLUMN {col_name} {col_def}"
                )
                if col_name == "value":
                    value_just_added = True
            except sqlite3.OperationalError:
                # Race with another connection that already added it. Safe.
                pass

        if value_just_added:
            self._backfill_fragment_values()

    def _backfill_fragment_values(self) -> None:
        """Compute initial ``value`` for every existing fragment.

        Runs once, when the column is added to a pre-existing DB. Reads each
        row's provenance + type + content, calls ``compute_fragment_value``,
        and updates the row. We don't touch ``updated_at`` because this is a
        schema-migration backfill, not a semantic change.
        """
        from .value import compute_fragment_value

        rows = self._conn.execute(
            "SELECT id, type, content, extraction_method, created_by_tool, metadata "
            "FROM fragments"
        ).fetchall()
        if not rows:
            return
        updates: list[tuple[float, str]] = []
        for row in rows:
            try:
                md = json.loads(row["metadata"]) if row["metadata"] else {}
            except (ValueError, TypeError):
                md = {}
            v = compute_fragment_value(
                type=row["type"],
                content=row["content"],
                extraction_method=row["extraction_method"] or "explicit",
                created_by_tool=row["created_by_tool"],
                metadata=md,
            )
            updates.append((v, row["id"]))
        # Single transaction so the backfill doesn't show partial state to
        # concurrent readers.
        with self._conn:
            self._conn.executemany(
                "UPDATE fragments SET value = ? WHERE id = ?",
                updates,
            )

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
        try:
            return self.create_identity(data)
        except sqlite3.IntegrityError:
            # Race: another thread/connection inserted the same handle
            # between our SELECT and INSERT. Re-query the canonical row.
            again = self.get_identity(data.handle)
            if again is not None:
                return again
            raise

    def list_identities(self, limit: int = 100, offset: int = 0) -> list[Identity]:
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
                    limit: int = 100, offset: int = 0) -> list[Scope]:
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

    def get_scope_lineage(self, handle: str) -> list[Scope]:
        """Return the scope and all its ancestors (self included), nearest first."""
        lineage: list[Scope] = []
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
        from .value import compute_fragment_value
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

        # Iter 31: dedupe shortcut for explicit writes. Compute a content
        # hash; if an identical fragment already exists in this scope, bump
        # its value (the "this matters again" signal) and return without
        # inserting a duplicate row. Passive-extracted fragments
        # (extraction_method != 'explicit') already dedupe via the inbox's
        # uq_candidates_dedup constraint, so we skip them here to keep that
        # path unchanged.
        method = (data.extraction_method or "explicit").lower()
        dedupe_key: Optional[str] = None
        if method == "explicit":
            import hashlib
            tool = (data.created_by_tool or "").lower()
            seed = f"{data.scope_id}|{data.type}|{tool}|{data.content}".encode("utf-8")
            dedupe_key = hashlib.sha256(seed).hexdigest()[:32]
            existing = self._conn.execute(
                "SELECT id FROM fragments WHERE dedupe_key = ? "
                "AND is_stale = 0 LIMIT 1",
                (dedupe_key,),
            ).fetchone()
            if existing:
                # Re-assert: nudge value + refresh updated_at, return cached.
                self._conn.execute(
                    "UPDATE fragments SET value = MIN(1.0, value + 0.05), "
                    "updated_at = ? WHERE id = ?",
                    (now, existing["id"]),
                )
                # Invalidate the recall cache for this scope so the bump
                # surfaces immediately.
                try:
                    from . import retrieval as _retr
                    _retr.invalidate_recall_cache(data.scope_id)
                except Exception:
                    pass
                hit = self.get_fragment(existing["id"])
                if hit is not None:
                    return hit

        # Iter 25 (Q-05 phases 1+2): score the fragment's recall-time value
        # from provenance + type + content. Persisted so retrieval can apply
        # it without a recompute per query.
        value = compute_fragment_value(
            type=data.type,
            content=data.content,
            extraction_method=data.extraction_method,
            created_by_tool=data.created_by_tool,
            metadata=data.metadata,
        )

        # Iter 31: soft-warn at write time when a fragment's content
        # exceeds 800 chars. Don't reject — but build the muscle memory
        # that fragments should be short. Long-form lives in Obsidian or
        # the commit body. The warning fires only for explicit writes
        # (passive extraction produces inherently varied content lengths
        # and is governed by its own confidence threshold).
        if method == "explicit" and len(data.content or "") > 800:
            logger.warning(
                "Fragment exceeds 800-char soft cap (len=%d, type=%s, tool=%s). "
                "Consider splitting into multiple fragments — long-form context "
                "lives better in Obsidian or commit bodies. First 80 chars: %r",
                len(data.content), data.type,
                data.created_by_tool or "unknown",
                (data.content or "")[:80],
            )

        self._conn.execute(
            """INSERT INTO fragments
               (id, type, content, scope_id, owner_id, confidence, version,
                ttl_seconds, expires_at, permanent, is_stale, stale_reason,
                tags, territory, source_commit_id, metadata, content_embedding,
                created_by_tool, created_in_session_id, created_against_commit,
                files_open_at_creation, supersedes_fragment_id,
                extraction_method, extraction_confidence, value, dedupe_key,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 0, NULL,
                       ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?)""",
            (frag_id, data.type, data.content, data.scope_id, data.owner_id,
             data.confidence, data.ttl_seconds, expires_at,
             1 if permanent else 0,
             json.dumps(data.tags), data.territory, commit_id,
             json.dumps(data.metadata), embedding,
             data.created_by_tool, data.created_in_session_id,
             data.created_against_commit,
             json.dumps(data.files_open_at_creation),
             data.supersedes_fragment_id,
             data.extraction_method, data.extraction_confidence, value,
             dedupe_key,
             now, now),
        )

        # Iter 31: invalidate the per-scope recall cache so the new write
        # surfaces immediately. Cheap (in-memory dict scan).
        try:
            from . import retrieval as _retr
            _retr.invalidate_recall_cache(data.scope_id)
        except Exception:
            pass

        # If this fragment supersedes another, keep the reverse pointer in sync
        # so future queries can walk the chain in either direction.
        if data.supersedes_fragment_id:
            self._conn.execute(
                "UPDATE fragments SET superseded_by_fragment_id = ? WHERE id = ?",
                (frag_id, data.supersedes_fragment_id),
            )

        return Fragment(
            id=frag_id, type=data.type, content=data.content,
            scope_id=data.scope_id, owner_id=data.owner_id,
            confidence=data.confidence, version=1,
            ttl_seconds=data.ttl_seconds, expires_at=expires_at,
            permanent=permanent, is_stale=False,
            tags=data.tags, territory=data.territory,
            source_commit_id=commit_id, metadata=data.metadata,
            created_by_tool=data.created_by_tool,
            created_in_session_id=data.created_in_session_id,
            created_against_commit=data.created_against_commit,
            files_open_at_creation=data.files_open_at_creation,
            supersedes_fragment_id=data.supersedes_fragment_id,
            extraction_method=data.extraction_method,
            extraction_confidence=data.extraction_confidence,
            value=value,
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

        updates: dict[str, Any] = {}
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

        # Iter 31: invalidate the recall cache for the updated row's scope
        # so the change surfaces immediately.
        try:
            from . import retrieval as _retr
            _retr.invalidate_recall_cache(row["scope_id"])
        except Exception:
            pass

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
                        offset: int = 0,
                        since: Optional[str] = None,
                        exclude_tool: Optional[str] = None) -> list[Fragment]:
        conditions = ["1=1"]
        params: list[Any] = []

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
        if since:
            conditions.append("created_at > ?")
            params.append(since)
        if exclude_tool:
            # NULL-tolerant: fragments without a recorded tool still count
            # as "not from <exclude_tool>".
            conditions.append("(created_by_tool IS NULL OR created_by_tool != ?)")
            params.append(exclude_tool)

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

    def peek_embedding_dimension(self) -> Optional[int]:
        """Return the dimension of the first non-null embedding in the DB, or None.

        Used at daemon startup (iter 23) to detect a provider/storage mismatch:
        if the active embedding provider's dimension differs from what's
        stored, recall results will be nonsense until `skein ingest . --reset`.
        Stored as raw float32 bytes, so dimension = len(bytes) / 4.
        """
        row = self._conn.execute(
            "SELECT content_embedding FROM fragments "
            "WHERE content_embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return len(row[0]) // 4

    def bump_recall_hits(self, fragment_ids: list[str]) -> None:
        """Iter 31 (Q-05 phase 3): single batched UPDATE bumping
        ``recall_hits`` and ``last_recalled_at`` for fragments that just
        landed in the top-K of a real recall response. Called by
        ``retrieval.recall`` after building results. Never raises — the
        caller wraps in try/except so a buggy telemetry call can never
        break recall.
        """
        if not fragment_ids:
            return
        placeholders = ",".join("?" * len(fragment_ids))
        self._conn.execute(
            f"""UPDATE fragments
                   SET recall_hits = recall_hits + 1,
                       last_recalled_at = datetime('now')
                 WHERE id IN ({placeholders})""",
            list(fragment_ids),
        )

    def record_recall_event(self, recall_id: str, query: str, scope_handle: str) -> None:
        """Iter 35: capture a recall event for outcome telemetry. INSERT OR
        IGNORE so daemon retries don't error; recall_id collision is a no-op."""
        self._conn.execute(
            "INSERT OR IGNORE INTO recall_events (recall_id, query, scope_handle) "
            "VALUES (?, ?, ?)",
            (recall_id, query[:500], scope_handle),
        )

    def link_recall_to_fragment(self, recall_id: str, fragment_id: str) -> bool:
        """Iter 35: link a recall event to a fragment written as a follow-up.
        Returns True if the recall_id was found and the link was created (or
        already existed); False if the recall_id is unknown. A single recall
        can link to many fragments — the LLM may write multiple notes from
        one round of recall."""
        row = self._conn.execute(
            "SELECT 1 FROM recall_events WHERE recall_id = ?", (recall_id,),
        ).fetchone()
        if not row:
            return False
        self._conn.execute(
            "INSERT OR IGNORE INTO recall_links (recall_id, fragment_id) "
            "VALUES (?, ?)",
            (recall_id, fragment_id),
        )
        return True

    def recall_write_stats(self, hours: int = 24) -> tuple[int, int]:
        """Iter 35: return ``(linked, total)`` for recall events in the last
        ``hours``. ``linked`` is the count of recall_ids with at least one
        fragment back-linked via ``from_recall``. Powers the doctor
        recall→write rate line."""
        cutoff = f"-{int(hours)} hours"
        total = self._conn.execute(
            "SELECT COUNT(*) FROM recall_events WHERE created_at >= datetime('now', ?)",
            (cutoff,),
        ).fetchone()[0]
        linked = self._conn.execute(
            "SELECT COUNT(DISTINCT re.recall_id) FROM recall_events re "
            "JOIN recall_links rl ON rl.recall_id = re.recall_id "
            "WHERE re.created_at >= datetime('now', ?)",
            (cutoff,),
        ).fetchone()[0]
        return (int(linked), int(total))

    def recent_writes_by_tool(self, hours: int = 24) -> dict[str, int]:
        """Iter 29 day-one: return ``{created_by_tool: count}`` for non-stale
        fragments written in the last ``hours``.

        Powers the cross-LLM activity line in the MCP ``initialize.instructions``
        greeting — the unique value prop ("cursor stored 3 decisions today, the
        next session you open in cursor will see your decisions") is most
        compelling when the LLM sees concrete numbers, not generic copy.
        """
        rows = self._conn.execute(
            """SELECT COALESCE(created_by_tool, 'unknown') AS tool, COUNT(*) AS c
               FROM fragments
               WHERE is_stale = 0
                 AND (expires_at IS NULL OR expires_at > datetime('now'))
                 AND created_at >= datetime('now', ?)
               GROUP BY tool ORDER BY c DESC""",
            (f"-{int(hours)} hours",),
        ).fetchall()
        return {row["tool"]: int(row["c"]) for row in rows}

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

    def count_fragments_by_type(
        self, scope_id: str, include_stale: bool = False,
    ) -> dict[str, int]:
        """Return ``{type: count}`` for a scope in a single SQL round-trip.

        Powers the `project_briefing` MCP tool / `/v1/briefing` endpoint.
        Default ``include_stale=False`` mirrors :py:meth:`count_fragments`.
        Types absent from the result simply don't appear in the dict — callers
        should default to 0 for any type they care about.
        """
        if include_stale:
            rows = self._conn.execute(
                "SELECT type, COUNT(*) AS c FROM fragments WHERE scope_id = ? "
                "GROUP BY type",
                (scope_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT type, COUNT(*) AS c FROM fragments WHERE scope_id = ? "
                "AND is_stale = 0 "
                "AND (expires_at IS NULL OR expires_at > datetime('now')) "
                "GROUP BY type",
                (scope_id,),
            ).fetchall()
        return {row["type"]: int(row["c"]) for row in rows}

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

    def get_fragments_without_embeddings(self, limit: int = 100) -> list[Fragment]:
        rows = self._conn.execute(
            "SELECT * FROM fragments WHERE content_embedding IS NULL "
            "AND is_stale = 0 ORDER BY created_at LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_fragment(r) for r in rows]

    # ------------------------------------------------------------------
    # Keyword search (FTS5 BM25)
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, scope_ids: list[str],
                        type_filter: Optional[list[str]] = None,
                        include_stale: bool = False,
                        limit: int = 20,
                        value_floor: float = 0.0) -> list[tuple[str, float]]:
        """Return (fragment_id, bm25_score) pairs ordered by relevance.

        BM25 score from FTS5 is negative (more negative = more relevant).
        We negate it so higher = better.

        Iter 31: ``value_floor`` skips low-value rows at the SQL layer
        (cheap range-scan on idx_fragments_value). Callers pass it
        through from retrieval.recall (0.15 by default; 0.0 when
        ``include_stale=True``).
        """
        if not scope_ids:
            return []

        placeholders = ",".join("?" * len(scope_ids))
        type_filter_clause = ""
        type_params: list[Any] = []
        if type_filter:
            tp = ",".join("?" * len(type_filter))
            type_filter_clause = f"AND f.type IN ({tp})"
            type_params = list(type_filter)

        stale_clause = "" if include_stale else (
            "AND f.is_stale = 0 "
            "AND (f.expires_at IS NULL OR f.expires_at > datetime('now'))"
        )
        value_clause = ""
        value_params: list[Any] = []
        if value_floor > 0.0:
            value_clause = "AND f.value >= ?"
            value_params = [float(value_floor)]

        rows = self._conn.execute(
            f"""
            SELECT fts.fragment_id, -bm25(fragments_fts) AS score
            FROM fragments_fts AS fts
            JOIN fragments AS f ON f.id = fts.fragment_id
            WHERE fragments_fts MATCH ?
              AND f.scope_id IN ({placeholders})
              {type_filter_clause}
              {stale_clause}
              {value_clause}
            ORDER BY bm25(fragments_fts)
            LIMIT ?
            """,
            [_fts_escape(query)] + scope_ids + type_params + value_params + [limit],
        ).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    # ------------------------------------------------------------------
    # Vector search (cosine, Python-side)
    # ------------------------------------------------------------------

    def vector_search(self, query_vec_bytes: bytes, scope_ids: list[str],
                       type_filter: Optional[list[str]] = None,
                       include_stale: bool = False,
                       limit: int = 20,
                       dimension: int = 768,
                       batch_size: int = 5000,
                       value_floor: float = 0.0) -> list[tuple[str, float]]:
        """Return (fragment_id, cosine_similarity) ordered by score.

        Vectorised: rows are pulled in ``batch_size`` chunks, the embedding
        BLOBs concatenated, reshaped into one float32 matrix, and scored in
        a single matmul. For 126 fragments × 384 dim this collapses ~80ms
        of per-row numpy churn down to <2ms. Memory bound is
        O(batch_size · dimension · 4 bytes) — fine on any laptop.
        """
        import heapq
        import numpy as np

        if not scope_ids:
            return []

        placeholders = ",".join("?" * len(scope_ids))
        type_filter_clause = ""
        type_params: list[Any] = []
        if type_filter:
            tp = ",".join("?" * len(type_filter))
            type_filter_clause = f"AND type IN ({tp})"
            type_params = list(type_filter)

        stale_clause = "" if include_stale else (
            "AND is_stale = 0 "
            "AND (expires_at IS NULL OR expires_at > datetime('now'))"
        )
        # Iter 31: SQL-level value floor — same rationale as keyword_search.
        # Particularly valuable here because each retained row costs
        # 4 × dimension bytes through Python's numpy reshape, so dropping
        # noise early compounds.
        value_clause = ""
        value_params: list[Any] = []
        if value_floor > 0.0:
            value_clause = "AND value >= ?"
            value_params = [float(value_floor)]

        query_vec = np.frombuffer(query_vec_bytes, dtype=np.float32)
        if len(query_vec) != dimension:
            return []
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm == 0.0:
            return []
        q_unit = query_vec / q_norm

        cur = self._conn.execute(
            f"""SELECT id, content_embedding FROM fragments
                WHERE content_embedding IS NOT NULL
                  AND scope_id IN ({placeholders})
                  {type_filter_clause}
                  {stale_clause}
                  {value_clause}""",
            scope_ids + type_params + value_params,
        )

        row_bytes = dimension * 4
        heap: list[tuple[float, str]] = []
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            ids = [r[0] for r in rows]
            blob = b"".join(r[1] for r in rows)
            # Drop any rows whose stored embedding is the wrong dimension
            # (e.g. legacy 768-dim fragments after switching providers): the
            # graceful path is to skip them rather than corrupt the matrix.
            if len(blob) != len(rows) * row_bytes:
                clean_ids: list[str] = []
                clean_bufs: list[bytes] = []
                for rid, rbuf in rows:
                    if rbuf is not None and len(rbuf) == row_bytes:
                        clean_ids.append(rid)
                        clean_bufs.append(rbuf)
                if not clean_ids:
                    continue
                ids = clean_ids
                blob = b"".join(clean_bufs)
            mat = np.frombuffer(blob, dtype=np.float32).reshape(-1, dimension)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = mat @ q_unit / norms
            for fid, s in zip(ids, sims.tolist()):
                if len(heap) < limit:
                    heapq.heappush(heap, (s, fid))
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, (s, fid))

        return sorted(((fid, float(s)) for s, fid in heap), key=lambda x: -x[1])

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
                      limit: int = 50, offset: int = 0) -> list[Commit]:
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
                     active_only: bool = True) -> list[Lease]:
        conditions = ["1=1"]
        params: list[Any] = []
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
                     limit: int = 50, offset: int = 0) -> list[Chunk]:
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

    def chunk_stats(self, scope_id: Optional[str] = None) -> dict[str, Any]:
        scope_clause = ""
        params: list[Any] = []
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

    def get_chunks_by_ids(self, ids: list[str]) -> dict[str, Chunk]:
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
        keys: list[tuple[str, int, int]],
    ) -> dict[tuple[str, int, int], str]:
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
        flat: list[Any] = [scope_id, source_root]
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
        self, query: str, scope_ids: list[str], *,
        languages: Optional[list[str]] = None,
        source_root: Optional[str] = None,
        limit: int = 30,
    ) -> list[tuple[str, float]]:
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
        self, query_vec_bytes: bytes, scope_ids: list[str], *,
        languages: Optional[list[str]] = None,
        source_root: Optional[str] = None,
        limit: int = 30,
        dimension: int = 768,
        batch_size: int = 5000,
    ) -> list[tuple[str, float]]:
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
        heap: list[tuple[float, str]] = []
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

    def stats(self) -> dict[str, int]:
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

    def get_fragments_by_ids(self, ids: list[str]) -> dict[str, Fragment]:
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
    # Provenance columns may be absent on rows from a pre-iter-14 DB even
    # after migration if the row isn't yet refreshed — defensively access.
    def _maybe(col: str, default=None):
        try:
            return row[col]
        except (IndexError, KeyError):
            return default

    files_open_raw = _maybe("files_open_at_creation", "[]") or "[]"
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
        created_by_tool=_maybe("created_by_tool"),
        created_in_session_id=_maybe("created_in_session_id"),
        created_against_commit=_maybe("created_against_commit"),
        files_open_at_creation=json.loads(files_open_raw),
        supersedes_fragment_id=_maybe("supersedes_fragment_id"),
        superseded_by_fragment_id=_maybe("superseded_by_fragment_id"),
        extraction_method=_maybe("extraction_method", "explicit") or "explicit",
        extraction_confidence=_maybe("extraction_confidence"),
        value=_maybe("value", 0.5) if _maybe("value", None) is not None else 0.5,
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
    def _segments(g: str) -> list[str]:
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
