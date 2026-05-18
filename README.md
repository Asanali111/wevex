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
# 1. Install (once) — package is `skn`, CLI is `skein`
pip install skn

# 2. Activate in any project
cd ~/Documents/your-project
skein up
```

The PyPI package is named **`skn`** because the natural guess `skein` is already taken by an unrelated Apache project. The CLI binary stays `skein` — install with `skn`, run with `skein`.

Other install paths that work the same:

```bash
pipx install skn          # recommended for CLI tools — isolated env, auto-PATH
uv tool install skn       # modern, fastest
pip3 install skn          # macOS users where `pip` points at Python 2
py -m pip install skn     # Windows
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

### After `skein up` — day-to-day usage

Skein is meant to be invisible. Most work happens through MCP tools your LLM already calls — `recall`, `remember`, `note_decision`, `search_code`, `boost`, `bury`, `archaeology`, `supersede` — so the user surface is small on purpose. The five CLI commands you'll actually reach for:

```bash
# See what's wired up
skein status

# Deep diagnostic (folds in the old `clients`, `events`, `chunks stats`,
# `projects list`, `daemon status`, value-distribution, inbox depth)
skein doctor

# What's the project state?
skein briefing
skein briefing --since 2d        # what changed in the last 2 days

# Live event stream
skein tail

# Interactive control panel
skein tui
```

`skein doctor` has two action flags that absorb what used to be standalone commands:

```bash
skein doctor --clean       # interactive cleanup (replaces `skein gc`)
skein doctor --reingest    # re-index the cwd (replaces `skein ingest`)
```

The daemon takes care of everything else automatically — AGENTS.md regen, TTL gc, docs sync, inbox auto-approve. You don't run a sync command; the file just stays current.

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

The visible surface is intentionally small — Skein is an invisible helper, not a CLI suite to learn. Ten commands cover everything a human needs:

```
skein up [path]    Start the daemon, register cwd, wire up detected clients
skein down         Stop daemon + watcher, uninstall hooks
skein restart      Restart the daemon
skein status       One-screen health: daemon, clients, fragment + chunk counts
skein doctor       Deep diagnostic. Flags: --clean / --reingest / --perf
skein tail         Live event stream
skein briefing     Project state. With --since <when>, becomes the diff feed
skein tui          Interactive control panel
skein config       View or set runtime configuration
skein connect      Wire installed LLM tools through Skein. --remove disconnects
```

What the daemon does automatically (no command, no maintenance):

- Regenerates `AGENTS.md` for each registered project when fragments change.
- Drains the extraction-candidate inbox: auto-approves above the confidence threshold, auto-rejects items older than 14 days that didn't clear.
- Sweeps expired TTLs (lease + fragment stale-mark).
- Tails docs (README/CHANGELOG/ADRs) into fragments via the docs watcher.

Need a fine-grained command from a previous version? Most still work but are hidden from `--help` and will be removed in a follow-up — prefer the MCP tool path (`recall`, `remember`, `note_decision`, `boost`, `bury`, `archaeology`, `supersede`) for anything an LLM session needs to do.

---

## MCP tools (for Claude Code, Cursor, Codex, etc.)

Once `skein up` runs and your client is connected, every MCP-capable LLM gets these tools:

| Tool | Description |
|---|---|
| `project_briefing(scope?)` | One-call project dashboard (300 tokens, <50ms) |
| `recall(query, scope, types?, limit?)` | Search for relevant context fragments |
| `recall_one(fragment_id)` | Full content of a specific fragment |
| `remember(content, type, scope, territory?, tags?)` | Store context |
| `note_decision(content, scope, alternatives?, rationale?)` | Record a decision with structure |
| `supersede(old_id, new_content, reason?, type?, tags?)` | Retire a fragment + create its replacement atomically |
| `boost(fragment_id, value?)` | Pin a fragment to high recall-value when the user says "this is important" |
| `bury(fragment_id)` | Floor a fragment's value when the user says "this is wrong" — kept in audit, hidden from recall |
| `archaeology(query, scope?, limit?)` | Reconstruct a decision's provenance (session, commit, supersede chain) |
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

- Run with sub-100ms latency on the warm path (no HTTP round trip)
- Continue working when the daemon isn't running
- Use whatever embedding provider is configured (auto-skip vector if none works)
- Cold-path (first hook firing after a long idle) is closer to 1 s due to Python interpreter + module imports; warm-up scripts in `~/.skein/cache/` mitigate this on macOS but it's not free

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

The default vector search streams chunk embeddings from SQLite in batches of 5,000 and computes cosine similarity with NumPy. On a modern laptop, measured numbers from `skein doctor --perf` on a 566-chunk repo: code search p50 ≈ 14 ms (worst-case 27 ms). The brute-force cosine is O(N) and starts to feel slow past ~50k chunks — for larger codebases, swap in [`sqlite-vec`](https://github.com/asg017/sqlite-vec) or [`usearch`](https://github.com/unum-cloud/usearch); both can drop in behind `Storage.chunks_vector_search` without schema changes.

Verify against your own DB with `skein doctor --perf`.

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
| `embedding_provider` | `bm25` (auto-detects `gemini` / `openai` if their API key is set) | `SKEIN_EMBEDDING_PROVIDER` |
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

444 tests (as of iter 15) covering storage, retrieval, REST API, MCP JSON-RPC, AGENTS.md renderer, sync (including Antigravity), the autonomous hook system, the codebase RAG layer (ingest + chunks search), the cross-platform daemon manager (launchd / systemd-user / nohup with TCC auto-relocate), scope auto-detection from git remotes, the `.skein/scope` pin honoring across all CLI commands, the active-projects registry, the watcher's incremental dispatch, the code scanner, the git-commit decision watcher, the bm25/gemini provider auto-detection, the `.git`-required guard on `skein up`, and `skein doctor --perf` self-measurement.

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
