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
  -- Provenance (iter 14.0) — every fragment carries the full origin story
  -- so `skein archaeology` can reconstruct decisions even years later.
  created_by_tool          TEXT,                 -- claude-code, cursor, codex, code-scanner, transcript-claude, …
  created_in_session_id    TEXT,                 -- session UUID from the originating client
  created_against_commit   TEXT,                 -- git rev-parse HEAD at creation time
  files_open_at_creation   TEXT NOT NULL DEFAULT '[]',  -- JSON array
  supersedes_fragment_id   TEXT REFERENCES fragments(id),
  superseded_by_fragment_id TEXT REFERENCES fragments(id),
  extraction_method        TEXT NOT NULL DEFAULT 'explicit',  -- explicit | code-scan | transcript-claude | …
  extraction_confidence    REAL,                 -- 1.0 for explicit; <1.0 for auto-extracted
  -- Iter 25 (Q-05 phase 1+2): cheap deterministic "is this fragment worth
  -- recalling?" score derived from provenance + type + content rubrics at
  -- write-time. Multiplied into the final RRF score in retrieval so noisy
  -- fragments fall to the bottom without being deleted. Range [0.05, 1.0];
  -- 0.5 is the neutral default for fragments created before the column
  -- existed.
  value            REAL NOT NULL DEFAULT 0.5,
  -- Iter 31 (Q-05 phase 3): behavioural value telemetry. Bumped by a
  -- single batched UPDATE inside retrieval.recall() when a fragment lands
  -- in the top-K of a real (non-cached) recall response. A daily
  -- background loop nudges `value` toward `base + 0.05*log(recall_hits)`
  -- so fragments that actually get used rise; unused ones fade. Skips
  -- pinned rows (value==1.0 + metadata.pinned==true).
  recall_hits      INTEGER NOT NULL DEFAULT 0,
  last_recalled_at TEXT,
  -- Iter 31: cheap content-hash dedupe for explicit writes
  -- (`remember`/`note_decision`). Re-asserting an identical fragment
  -- bumps its value and refreshes updated_at instead of inserting a
  -- duplicate row. NULL for passive-extracted fragments (they go
  -- through inbox auto-approve which has its own dedupe table).
  dedupe_key       TEXT,
  created_at       TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fragments_scope    ON fragments(scope_id);
CREATE INDEX IF NOT EXISTS idx_fragments_type     ON fragments(type);
CREATE INDEX IF NOT EXISTS idx_fragments_expires  ON fragments(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fragments_stale    ON fragments(is_stale);
CREATE INDEX IF NOT EXISTS idx_fragments_territory ON fragments(territory) WHERE territory IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fragments_supersedes ON fragments(supersedes_fragment_id) WHERE supersedes_fragment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fragments_superseded_by ON fragments(superseded_by_fragment_id) WHERE superseded_by_fragment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fragments_tool     ON fragments(created_by_tool) WHERE created_by_tool IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_fragments_method   ON fragments(extraction_method);
CREATE INDEX IF NOT EXISTS idx_fragments_value    ON fragments(value);
-- Iter 31: dedupe lookup must be O(log n) so the create_fragment
-- shortcut is cheap. Partial index — only explicit writes get a key.
CREATE INDEX IF NOT EXISTS idx_fragments_dedupe   ON fragments(dedupe_key) WHERE dedupe_key IS NOT NULL;
-- Iter 31: behavioural-value decay loop scans the recently-recalled
-- subset. Partial index keeps it tiny.
CREATE INDEX IF NOT EXISTS idx_fragments_recalled ON fragments(last_recalled_at) WHERE last_recalled_at IS NOT NULL;

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


-- ---------------------------------------------------------------------------
-- MCP client tokens (iter 14.0)
-- One row per LLM client connected to Skein. `skein up` populates this so
-- every MCP request can be attributed to the originating tool
-- (claude-code / cursor / codex / …).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mcp_clients (
  token_prefix     TEXT PRIMARY KEY,           -- first 16 chars of bearer token (unique enough)
  client_name      TEXT NOT NULL,              -- canonical lowercase: claude-code, cursor, codex, ...
  display_name     TEXT,                       -- user-facing label
  full_token_hash  TEXT,                       -- optional: bcrypt for full verification later
  created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_mcp_clients_name ON mcp_clients(client_name);

-- ---------------------------------------------------------------------------
-- Extraction candidates (iter 14.2)
-- Pending fragments produced by the passive watchers (code scanner +
-- transcript extractor). Medium-confidence candidates land here for
-- `skein inbox` review before being promoted to real fragments.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS extraction_candidates (
  id                  TEXT PRIMARY KEY,
  scope_id            TEXT NOT NULL REFERENCES scopes(id) ON DELETE CASCADE,
  content             TEXT NOT NULL,
  type                TEXT NOT NULL CHECK (type IN (
                        'preference','fact','decision','state',
                        'observation','requirement','procedure','conversation')),
  territory           TEXT,
  tags                TEXT NOT NULL DEFAULT '[]',     -- JSON array
  confidence          REAL NOT NULL,
  source_tool         TEXT NOT NULL,                  -- code-scanner / transcript-claude / ...
  source_session_id   TEXT,
  source_file         TEXT,                           -- when extracted from a file
  source_message_ts   TEXT,                           -- when extracted from a chat message
  status              TEXT NOT NULL DEFAULT 'pending',-- pending | approved | rejected
  reviewed_at         TEXT,
  promoted_fragment_id TEXT REFERENCES fragments(id), -- non-null after approval
  created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_candidates_status ON extraction_candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_scope ON extraction_candidates(scope_id);
CREATE INDEX IF NOT EXISTS idx_candidates_tool ON extraction_candidates(source_tool);

-- Dedupe: same content + scope + source_file shouldn't pile up
CREATE UNIQUE INDEX IF NOT EXISTS uq_candidates_dedup
  ON extraction_candidates(scope_id, content, source_tool);

-- ---------------------------------------------------------------------------
-- AGENTS.md render state (iter 26 / ADR-002)
-- Daemon auto-sync watches the fragment set for changes and regenerates the
-- per-project AGENTS.md when the rendered output's hash differs from
-- whatever was last written. Replaces the manual `skein sync` command.
-- One row per (scope, on-disk path). last_render_hash is sha256 over the
-- exact bytes written to disk; mismatch = "fragments changed, regen".
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agents_md_state (
  scope_handle      TEXT NOT NULL,
  file_path         TEXT NOT NULL,
  last_render_hash  TEXT NOT NULL,
  last_render_at    TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (scope_handle, file_path)
);

-- ---------------------------------------------------------------------------
-- Transcript cursors (iter 14.2)
-- Track how far into each Claude Code transcript JSONL we've read, so the
-- watcher can resume cleanly after daemon restart without re-extracting
-- everything.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transcript_cursors (
  file_path        TEXT PRIMARY KEY,
  last_byte_offset INTEGER NOT NULL DEFAULT 0,
  last_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
  client_name      TEXT NOT NULL                  -- claude-code, ...
);
