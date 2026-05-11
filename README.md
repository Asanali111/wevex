# Skein

> **Local MCP context bus for coding LLMs.**  
> One daemon, every coding client connected to the same typed context.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The problem

Every LLM tool ships its own memory silo. Switching tools = starting from zero. Two agents on the same project can't see each other's work. Copy-paste between Claude Code, Cursor, Codex, and Gemini CLI is the actual current solution.

## The solution

One local daemon. Every coding client connects via MCP (or AGENTS.md for non-MCP tools). They share typed **fragments** (decisions, state, observations, requirements) per **scope** (project / team / org), coordinate via **advisory leases**, and all get the same rendered **AGENTS.md**.

```
  Claude Code  Cursor  Codex  Gemini CLI  VS Code  Copilot  opencode
       │           │        │       │          │        │        │
       └──── MCP Streamable HTTP (127.0.0.1:8765/mcp) ─────────┘
                                    │
                       ┌────────────┴────────────┐
                       │   FastAPI + MCP server  │  one process, one port
                       └────────────┬────────────┘
                                    │
                          SQLite + FTS5 + numpy
                    (fragments · commits · leases · scopes)
                                    │
              ┌──────────────────┬──┴──────────────────┐
              │   CLI (skein)    │                      │
              │  init/serve/sync │    AGENTS.md (file)  │
              │  remember/recall │    Pi · OpenClaw     │
              │  lease/doctor    │    (non-MCP fallback) │
              └──────────────────┴─────────────────────┘
```

---

## Quick start

**Two commands.** That's it.

```bash
# 1. Install (once)
curl -fsSL https://raw.githubusercontent.com/ameliomar/skein/main/bin/install.sh | sh

# 2. Activate in any project
cd ~/Documents/your-project
skein up
```

After `skein up`, every connected LLM (Claude Code, Cursor, Codex, Gemini CLI, Antigravity, Copilot, VS Code, opencode) automatically has shared context for the project. The daemon runs as a background service that survives terminal close and reboots (launchd on macOS, systemd-user on Linux, nohup elsewhere).

`skein up` is **idempotent** — safe to run repeatedly. It does:

| | |
|---|---|
| 1. Init | Generate bearer token + config (`~/.config/skein/config.json`) on first run |
| 2. Daemon | Install + start a background service (auto-relocates the venv to `~/.skein/venv` if needed for macOS TCC) |
| 3. Scope | Auto-detect from git remote (`git@github.com:user/repo.git` → `project:repo`) or cwd name |
| 4. Hooks | Drop `.claude/settings.json` + `.cursor/rules/skein.mdc` + `.skein/scope` so every LLM auto-recalls and auto-remembers |
| 5. Sync | Write MCP configs for every detected client (Cursor, VS Code, Codex, Gemini CLI, Antigravity, opencode, Claude Code) + AGENTS.md + CLAUDE.md |
| 6. Ingest | Index the codebase for semantic search — incremental, free re-runs |
| 7. Watcher | Spawn a session-scoped subprocess that re-ingests changed files within ~2 seconds. Survives terminal close; dies on logout (re-spawned by next `skein up`) |

To turn it off:

```bash
skein down       # stop the daemon and remove hooks from this project
skein restart    # restart the daemon
skein daemon status   # see what's running
skein daemon logs     # tail the daemon log
```

### After `skein up` — explore the data

```bash
# Store typed context
skein remember "use Redis for session caching" --type decision
skein note "use PostgreSQL as primary DB" \
    --alternatives "MySQL, SQLite" \
    --rationale "better JSON support and pgvector extension"
skein remember "API rate limit is 1000 req/min" --type fact --territory backend/api

# Search typed context
skein recall "caching strategy"
skein recall "database" --type decision --limit 5

# Search the codebase
skein search "how does authentication work"
skein search "rate limiting" --language python

# Multi-agent coordination
skein lease "backend/auth/**" --reason "refactoring auth"
skein leases

# Diagnose
skein doctor
```

---

## Fragment types

| Type | Default TTL | When to use |
|---|---|---|
| `preference` | 90 days | "prefer async/await over callbacks" |
| `fact` | 30 days | "API rate limit is 1000 req/min" |
| `decision` | 30 days | "use Redis for caching layer" |
| `state` | 7 days | "API schema v3 is current" |
| `observation` | 14 days | "auth middleware has a race condition" |
| `requirement` | permanent | "users must be able to export their data" |
| `procedure` | permanent | runbooks, how-tos |
| `conversation` | 30 days | extracted from chat threads |

---

## CLI reference

```
skein up                  One-command bootstrap: init + daemon + hooks + sync + ingest + watcher
skein down                Stop daemon, kill watcher, uninstall hooks from this project
skein restart             Restart the persistent daemon
skein watch <path>        Foreground watcher (called by skein up; debug-friendly)
skein projects list       Active projects + watcher status
skein projects remove     Unregister a project
skein daemon status       Show daemon backend, PID, and health
skein daemon logs         Tail ~/.config/skein/logs/daemon.{out,err}
skein init                Lower-level: generate token, create config (skein up does this)
skein serve               Lower-level: run the daemon in foreground
skein sync                Lower-level: only write MCP configs (no daemon, no hooks)
skein hooks install       Lower-level: only install hooks (no daemon, no sync)
skein hooks list          Show installed hooks (project + global)
skein hooks uninstall     Remove Skein-managed hooks (keeps user-added)
skein remember            Store a fragment
skein recall              Search fragments (hybrid BM25 + vector + RRF)
skein note                Record a decision (--alternatives / --rationale)
skein ingest <path>       Index a directory of code/docs for RAG
skein search "<query>"    Hybrid BM25+vector search over indexed code
skein chunks stats        Show how much code is indexed
skein chunks list         List indexed chunks for inspection
skein chunks delete-root  Delete all chunks under a source_root
skein lease               Acquire an advisory lease on a file-glob
skein leases              List active leases
skein agents-md           Print/write the rendered AGENTS.md
skein status              Show daemon stats
skein doctor              Diagnose config issues across all sync targets
skein config show         Print current config (token redacted)
skein config set          Update a config key
skein scope create        Create a new scope
skein scope list          List all scopes
```

Every command accepts `--json` for machine-readable output.

---

## MCP tools (for Claude Code, Cursor, Codex, etc.)

Once `skein sync` runs, every MCP-capable client gets these tools:

| Tool | Description |
|---|---|
| `recall(query, scope, types?, limit?)` | Search for relevant context fragments |
| `recall_one(fragment_id)` | Full content of a specific fragment |
| `remember(content, type, scope, territory?, tags?)` | Store context |
| `note_decision(content, scope, alternatives?, rationale?)` | Record a decision |
| `search_code(query, scope, languages?, source_root?, limit?)` | Hybrid search over the ingested codebase |
| `claim_lease(glob, scope, ttl_seconds?)` | Advisory lock on file-glob |
| `release_lease(lease_id)` | Release a lease |
| `query_leases(scope?)` | List active leases |

MCP resources (auto-injected):

| Resource URI | Content |
|---|---|
| `context://{scope}/state` | Current state fragments |
| `context://{scope}/decisions` | All active decisions |
| `context://{scope}/agents-md` | Rendered AGENTS.md |
| `context://{scope}/recent-commits` | Last 20 commits |

MCP prompt:

| Prompt | Description |
|---|---|
| `session_start` | Auto-inject AGENTS.md + 5 relevant fragments at session start |

---

## Autonomous mode

`skein hooks install` turns Skein from "tools the LLM can call" into "context that flows automatically without anyone asking." It writes a small set of files into the current project:

| File | Purpose |
|---|---|
| `.skein/scope` | Pins which scope hooks should use for this project |
| `.claude/settings.json` | Registers `SessionStart`, `UserPromptSubmit`, `Stop`, `PostToolUse` hooks |
| `.cursor/rules/skein.mdc` | Auto-applied Cursor rule that tells Composer to call `recall`/`remember` proactively |

What each hook does:

| Hook | Fires on | Effect |
|---|---|---|
| `SessionStart` | Claude Code session opens | Injects the most-relevant `requirement` / `decision` / `state` fragments into the session as context |
| `UserPromptSubmit` | Every user prompt to Claude Code | Recalls fragments scoring above a threshold for the prompt and injects them |
| `Stop` | Claude finishes a turn | Scans the assistant message for decision-shaped sentences (e.g. *"I decided to use FastAPI"*) and persists them as `decision` fragments tagged `auto-extracted` |
| `PostToolUse` | After `Edit` / `Write` / `MultiEdit` | Records the file path as an `observation` fragment with the territory (e.g. `backend/auth`) |

The hooks talk to the SQLite DB **directly**, so they:

- Run with sub-100ms latency (no HTTP round trip)
- Continue working when the daemon isn't running
- Use whatever embedding provider is configured (auto-skip vector if none works)

**Multi-tool scenario:** Open Claude Code in a hook-installed project and ask it to make a decision. Close it. Open Cursor in the same project — it reads `AGENTS.md` (rendered from the same fragments) and reads its rule file (which tells it to `recall` proactively). Cursor sees the decision Claude just made.

```bash
# After init + serve + sync, run once per project:
cd ~/Documents/your-project
skein hooks install --scope project:your-project

# Verify what was installed
skein hooks list

# Remove later (preserves any user-added hook entries):
skein hooks uninstall
```

`--global` adds the same hooks at `~/.claude/settings.json` so they apply across every project (the per-project `.skein/scope` file pins which scope to use).

---

## Codebase RAG

Fragments are great for typed context — decisions, requirements, observations. They're not great for "find the function that does X across this 50k-LOC codebase." For that, Skein has a separate **chunks** layer.

### Ingest a codebase

```bash
# Index every supported file under ./src
skein ingest ./src --scope project:myapp

# Filter by extension
skein ingest . --include .py,.md --scope project:myapp

# Re-index after changes (skips unchanged chunks via content hash)
skein ingest . --scope project:myapp

# Re-index and remove chunks for files that no longer exist
skein ingest . --scope project:myapp --prune

# Wipe and rebuild
skein ingest . --scope project:myapp --reset
```

What gets ingested:

- **Languages auto-detected** by extension: Python, JS, TS, Go, Rust, Java, Kotlin, Swift, Ruby, PHP, C, C++, C#, Scala, Clojure, Elixir, Haskell, Lua, Shell, SQL, Markdown, YAML, JSON, TOML, HTML, CSS, Dockerfile, Terraform, Protobuf, GraphQL, Vue, Svelte, Dart, R, …
- **Excluded by default**: `.git`, `node_modules`, `__pycache__`, `venv`, `.venv`, `dist`, `build`, `target`, `.next`, `.cache`, `_archive_v2`, …
- **Skipped**: files larger than 512 KB (override with `--max-bytes`), binary files, anything that fails UTF-8 decode

Each file is split into overlapping line windows (default 80 lines, 10-line overlap). Each chunk is hashed with SHA-256 — re-ingesting a file whose contents haven't changed is **free** (no DB write, no re-embedding API call).

### Search the ingested code

```bash
# CLI
skein search "how does authentication work"
skein search "rate limit middleware" --language python --limit 5
skein search "store fragment with embedding" --root skein

# REST
curl -X POST http://127.0.0.1:8765/v1/chunks/search \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query":"auth bearer","scope":"project:myapp","limit":5}'

# MCP — Claude Code, Cursor, etc. call this directly
search_code(query="how does auth work", scope="project:myapp")
```

Results return file path + line range + the matched chunk content, ranked by hybrid BM25 + vector + RRF.

### Stats / inspection

```bash
skein chunks stats --scope project:myapp
# → 107 chunks across 24 files
#   By language: python (104), sql (3)
#   By root: myapp (107)

skein chunks list --scope project:myapp --language python
skein chunks delete-root old-experiment --scope project:myapp
```

### Why chunks separate from fragments?

| Layer | Use case | Typical count | What you put there |
|---|---|---|---|
| **Fragments** | Typed atomic context (the bus) | 100s–10ks | Decisions, requirements, preferences, observations |
| **Chunks** | Codebase / document RAG | 1k–1M | Code files, docs, READMEs, ADRs |

Both share the same scope hierarchy and hybrid-retrieval pipeline, but they're indexed separately so a "what was the auth decision?" query (recall) doesn't compete with "show me code that touches auth" (search).

### Scaling notes

The default vector search streams chunk embeddings from SQLite in batches of 5,000 and computes cosine similarity with NumPy. On a modern laptop this is ≤200 ms for 50k chunks at 768 dim — comfortably fast for a single project. For million-chunk codebases, swap in [`sqlite-vec`](https://github.com/asg017/sqlite-vec) or [`usearch`](https://github.com/unum-cloud/usearch) — both can drop in behind `Storage.chunks_vector_search` without schema changes.

---

## REST API

```
GET  /health                    Public health check + stats
POST /v1/scopes                 Create a scope
GET  /v1/scopes                 List scopes
GET  /v1/scopes/{handle}/lineage Scope hierarchy
POST /v1/fragments              Create a fragment (auto-embeds, auto-commits)
GET  /v1/fragments              List fragments (filter by scope/type)
GET  /v1/fragments/search       Keyword+semantic search (GET)
POST /v1/fragments/recall       Hybrid search (POST with full RecallRequest)
GET  /v1/fragments/{id}         Get fragment
PATCH /v1/fragments/{id}        Update with OCC (send expected_version)
DELETE /v1/fragments/{id}       Soft-delete
GET  /v1/commits                Commit log
GET  /v1/commits/{id}           Single commit
POST /v1/leases                 Acquire advisory lease
GET  /v1/leases                 List leases
DELETE /v1/leases/{id}          Release lease
POST /v1/chunks/search          Codebase RAG hybrid search
GET  /v1/chunks/search          Same, GET form
GET  /v1/chunks                 List indexed chunks
GET  /v1/chunks/stats           Per-scope chunk counts by language/root
DELETE /v1/chunks/{root}        Delete every chunk under a source_root
POST /mcp                       MCP JSON-RPC endpoint
```

Interactive docs at `http://127.0.0.1:8765/docs`.

---

## Architecture decisions

| Decision | Chosen | Alternatives | Reason |
|---|---|---|---|
| Storage | SQLite + FTS5 + numpy | Postgres + pgvector | Zero external deps; portability |
| Retrieval | Hybrid BM25 + vector + RRF | Vector-only | +20% recall@10 |
| MCP transport | Streamable HTTP (hand-rolled) | `mcp[cli]` SDK | Python 3.9 compat; minimal deps |
| Auth | Local bearer token | OAuth 2.1 | Sufficient for local-first v1 |
| Embeddings | `hash` (offline, deterministic) | Gemini, OpenAI | Works with zero API keys; swap via config |
| Instruction file | AGENTS.md (canonical) | CLAUDE.md per-tool | ~60k repos; every major client reads it |
| Coordination | Advisory leases + OCC | CRDTs | Right abstraction for semantic conflicts |

See `20 Projects/Company Brain/Skein - Pivot ADR Log.md` for full ADR history.

---

## Configuration

Config file: `~/.config/skein/config.json` (created by `skein init`).

| Key | Default | Env override |
|---|---|---|
| `port` | 8765 | `SKEIN_PORT` |
| `host` | 127.0.0.1 | `SKEIN_HOST` |
| `db_path` | `~/.config/skein/skein.db` | `SKEIN_DB_PATH` |
| `embedding_provider` | `hash` | `SKEIN_EMBEDDING_PROVIDER` |
| `bearer_token` | (generated) | `SKEIN_BEARER_TOKEN` |
| `default_scope` | `project:default` | `SKEIN_DEFAULT_SCOPE` |

Embedding providers: `hash` (offline), `gemini` (requires `GEMINI_API_KEY`), `openai` (requires `OPENAI_API_KEY`).

---

## Running tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest tests/ -v
```

229 tests covering storage, retrieval, REST API, MCP JSON-RPC, AGENTS.md renderer, sync (including Antigravity), the autonomous hook system, the codebase RAG layer (ingest + chunks search), the cross-platform daemon manager (launchd / systemd-user / nohup with TCC auto-relocate), scope auto-detection from git remotes, the `.skein/scope` pin honoring across all CLI commands, the active-projects registry, the watcher's incremental dispatch (insert/update/prune on file changes + deletes), and the watcher-process manager (session-scoped subprocess spawn / kill / liveness checks).

---

## Scopes

Scopes form a visibility hierarchy: `public ⊃ org ⊃ team ⊃ project ⊃ personal`.  
A recall query on `project:foo` returns fragments from that project, its team, org, and public.

```bash
# Create a scope hierarchy
skein scope create org:acme --name "Acme Corp"
skein scope create team:backend --parent org:acme
skein scope create project:api-service --parent team:backend
```

---

## Embedding quality

The `hash` provider (default) is **not semantically meaningful** — it exists so Skein boots with zero API keys and tests run offline. Retrieval still works via BM25 (keyword). For semantic retrieval, configure Gemini:

```bash
export GEMINI_API_KEY=your-key
skein init --embedding-provider gemini
```

---

## License

MIT — see [LICENSE](LICENSE).
