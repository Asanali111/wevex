-- Skein SQLite schema v1
-- Uses FTS5 for keyword search and stores raw embedding floats in a BLOB column.
-- Vector cosine similarity is done Python-side (retrieval.py) using numpy.
-- All tables use TEXT UUIDs (gen'd Python-side via uuid4).

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Identities
-- Every entity that touches Skein (user, agent, LLM process) gets a row here.
-- handle: unique, namespaced — "user:ameliomar", "agent:cursor:proj-abc"
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS identities (
  id         TEXT PRIMARY KEY,
  handle     TEXT UNIQUE NOT NULL,
  type       TEXT NOT NULL CHECK (type IN ('user','agent','llm','service')),
  name       TEXT NOT NULL,
  config     TEXT NOT NULL DEFAULT '{}',   -- JSON blob
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- Scopes  (visibility hierarchy)
-- public ⊃ org:<n> ⊃ team:<n> ⊃ project:<n> ⊃ personal:<user>
-- Querying a scope returns its own fragments PLUS all ancestors.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scopes (
  id              TEXT PRIMARY KEY,
  handle          TEXT UNIQUE NOT NULL,
  type            TEXT NOT NULL CHECK (type IN ('public','org','team','project','personal')),
  name            TEXT NOT NULL,
  parent_scope_id TEXT REFERENCES scopes(id) ON DELETE CASCADE,
  owner_id        TEXT NOT NULL REFERENCES identities(id),
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scope_memberships (
  id          TEXT PRIMARY KEY,
  scope_id    TEXT NOT NULL REFERENCES scopes(id) ON DELETE CASCADE,
  identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
  role        TEXT NOT NULL CHECK (role IN ('owner','admin','contributor','viewer')),
  granted_at  TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(scope_id, identity_id)
);

-- ---------------------------------------------------------------------------
-- Commits  (append-only audit log)
-- Every mutation creates a commit row. This is written BEFORE the fragment
-- rows it references so foreign keys point forward if needed (relaxed here).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS commits (
  id                 TEXT PRIMARY KEY,
  author_id          TEXT NOT NULL REFERENCES identities(id),
  scope_id           TEXT NOT NULL REFERENCES scopes(id),
  parent_commit_id   TEXT REFERENCES commits(id),
  message            TEXT NOT NULL,
  fragments_added    TEXT NOT NULL DEFAULT '[]',    -- JSON array of UUIDs
  fragments_modified TEXT NOT NULL DEFAULT '[]',
  fragments_removed  TEXT NOT NULL DEFAULT '[]',
  metadata           TEXT NOT NULL DEFAULT '{}',
  created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_commits_scope     ON commits(scope_id);
CREATE INDEX IF NOT EXISTS idx_commits_created   ON commits(created_at);

-- ---------------------------------------------------------------------------
-- Fragments  (the atomic unit of context)
-- content_embedding: raw float32 bytes (numpy .tobytes()) — NULL until embedded.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fragments (
  id               TEXT PRIMARY KEY,
  type             TEXT NOT NULL CHECK (type IN (
                     'preference','fact','decision','state',
                     'observation','requirement','procedure','conversation')),
  content          TEXT NOT NULL,
  scope_id         TEXT NOT NULL REFERENCES scopes(id) ON DELETE CASCADE,
  owner_id         TEXT NOT NULL REFERENCES identities(id),
  confidence       REAL,
  version          INTEGER NOT NULL DEFAULT 1,
  ttl_seconds      INTEGER,
  expires_at       TEXT,           -- ISO datetime or NULL
  permanent        INTEGER NOT NULL DEFAULT 0,   -- boolean
  is_stale         INTEGER NOT NULL DEFAULT 0,
  stale_reason     TEXT,
  tags             TEXT NOT NULL DEFAULT '[]',   -- JSON array of strings
  territory        TEXT,                         -- "backend/auth", "frontend/ui"
  source_commit_id TEXT REFERENCES commits(id),
  metadata         TEXT NOT NULL DEFAULT '{}',
  content_embedding BLOB,                        -- NULL until embedded
  created_at       TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fragments_scope    ON fragments(scope_id);
CREATE INDEX IF NOT EXISTS idx_fragments_type     ON fragments(type);
CREATE INDEX IF NOT EXISTS idx_fragments_expires  ON fragments(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fragments_stale    ON fragments(is_stale);
CREATE INDEX IF NOT EXISTS idx_fragments_territory ON fragments(territory) WHERE territory IS NOT NULL;

-- FTS5 virtual table for full-text (BM25-like) search
CREATE VIRTUAL TABLE IF NOT EXISTS fragments_fts
  USING fts5(
    content,
    fragment_id UNINDEXED,   -- carry the real PK through for joins
    tokenize = 'porter ascii'
  );

-- Triggers to keep FTS in sync with fragments
CREATE TRIGGER IF NOT EXISTS fragments_fts_insert
  AFTER INSERT ON fragments BEGIN
    INSERT INTO fragments_fts(content, fragment_id) VALUES (new.content, new.id);
  END;

CREATE TRIGGER IF NOT EXISTS fragments_fts_delete
  AFTER DELETE ON fragments BEGIN
    DELETE FROM fragments_fts WHERE fragment_id = old.id;
  END;

CREATE TRIGGER IF NOT EXISTS fragments_fts_update
  AFTER UPDATE OF content ON fragments BEGIN
    DELETE FROM fragments_fts WHERE fragment_id = old.id;
    INSERT INTO fragments_fts(content, fragment_id) VALUES (new.content, new.id);
  END;

-- Auto-update updated_at on fragment modification
CREATE TRIGGER IF NOT EXISTS fragments_updated_at
  AFTER UPDATE ON fragments BEGIN
    UPDATE fragments SET updated_at = datetime('now') WHERE id = new.id;
  END;

-- ---------------------------------------------------------------------------
-- Leases  (advisory file-glob locks with TTL)
-- expires_at is an ISO datetime string; cleanup removes expired rows.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS leases (
  id          TEXT PRIMARY KEY,
  scope_id    TEXT NOT NULL REFERENCES scopes(id) ON DELETE CASCADE,
  glob        TEXT NOT NULL,           -- "backend/auth/**"
  owner_id    TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
  reason      TEXT,
  acquired_at TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at  TEXT NOT NULL,
  metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_leases_scope   ON leases(scope_id);
CREATE INDEX IF NOT EXISTS idx_leases_expires ON leases(expires_at);
CREATE INDEX IF NOT EXISTS idx_leases_owner   ON leases(owner_id);

-- ---------------------------------------------------------------------------
-- Chunks  (codebase / document RAG layer)
--
-- Distinct from `fragments` so that recall over typed context (decisions,
-- requirements, …) is not polluted with millions of code lines.  Search
-- happens via the dedicated `search_code` / chunks/search endpoints.
--
--   source_root: a stable label for the ingest base (usually a project
--                directory). Used for incremental re-ingest and bulk delete.
--   source_path: relative path within source_root (forward slashes).
--   content_hash: sha256 of the chunk content. Lets `skein ingest` skip
--                 unchanged chunks on repeat runs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
  id                TEXT PRIMARY KEY,
  scope_id          TEXT NOT NULL REFERENCES scopes(id) ON DELETE CASCADE,
  source_root       TEXT NOT NULL,
  source_path       TEXT NOT NULL,
  language          TEXT,
  chunk_type        TEXT NOT NULL DEFAULT 'window',  -- window | section | file | symbol
  symbol_name       TEXT,
  line_start        INTEGER NOT NULL,
  line_end          INTEGER NOT NULL,
  content           TEXT NOT NULL,
  content_hash      TEXT NOT NULL,
  content_embedding BLOB,
  metadata          TEXT NOT NULL DEFAULT '{}',
  created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(scope_id);
CREATE INDEX IF NOT EXISTS idx_chunks_root  ON chunks(scope_id, source_root);
CREATE INDEX IF NOT EXISTS idx_chunks_path  ON chunks(scope_id, source_root, source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash  ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_lang  ON chunks(language);

-- One chunk per (scope, root, path, line range) — guards against accidental dupes
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_location
  ON chunks(scope_id, source_root, source_path, line_start, line_end);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
  USING fts5(
    content,
    chunk_id UNINDEXED,
    tokenize = 'porter ascii'
  );

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
  AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(content, chunk_id) VALUES (new.content, new.id);
  END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
  AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
  END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update
  AFTER UPDATE OF content ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
    INSERT INTO chunks_fts(content, chunk_id) VALUES (new.content, new.id);
  END;
