# Skein

> **Local MCP context bus for coding LLMs.**  
> One daemon, every coding client connected to the same typed context.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## The problem

Every LLM tool ships its own memory silo. Switching tools = starting from zero. Two agents on the same project can't see each other's work. Copy-paste between Claude Code, Cursor, Windsurf, Hermes, and a dozen other tools is the actual current solution.

## The solution

One local daemon. Every coding client connects via MCP (or AGENTS.md for non-MCP tools). They share typed **fragments** (decisions, state, observations, requirements) per **scope** (project / team / org), coordinate via **advisory leases**, and all get the same rendered **AGENTS.md**.

```
  Claude Code  Cursor  Windsurf  Kiro  Codex  Hermes  Goose  Crush  gptme
       │           │        │      │      │       │       │      │      │
       └──────────── MCP Streamable HTTP (127.0.0.1:8765/mcp) ─────────┘
  Continue.dev  Antigravity  VS Code / Copilot  opencode  (+ more via skein connect)
       │               │              │              │
       └───────────────┴──────────────┴──────────────┘
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

After `skein up`, every detected client automatically has shared context for the project. The daemon runs as a background service that survives terminal close **and reboots** on all three OSes — launchd agent on macOS, systemd-user unit on Linux, Scheduled Task (logon trigger, restart-on-failure) on Windows.

### Supported clients

`skein connect` auto-detects and wires up any of these:

| Client | MCP config written | Notes |
|--------|--------------------|-------|
| Claude Code | `claude mcp add skein` | Via CLI; bearer token in header |
| Cursor | `.cursor/mcp.json` | |
| Windsurf | `.windsurf/mcp.json` | Uses `serverUrl` key |
| Kiro | `.kiro/settings/mcp.json` | AWS spec-first IDE (GA May 2026) |
| VS Code / Copilot | `.vscode/mcp.json` | One entry covers both |
| Codex CLI | `.codex/config.toml` | TOML `[[mcpServers]]` block |
| Antigravity | `~/.gemini/antigravity/mcp_config.json` | Google's Gemini CLI replacement |
| Hermes | `~/.hermes/config.yaml` + `~/.hermes/.env` | Nous Research; token in env var |
| Goose | `~/.config/goose/config.yaml` | Block; `streamable_http` transport |
| Crush | `.crush.json` | Charm; explicit `"type": "http"` |
| gptme | `~/.config/gptme/config.toml` | TOML `[[mcp.servers]]` block |
| Continue.dev | `~/.continue/mcpServers/skein.yaml` | Dedicated block file |
| opencode | `~/.config/opencode/config.json` | |
| Gemini CLI | `~/.gemini/settings.json` | Sunset June 18 2026; use Antigravity |

**Pi.dev**: no native MCP support (deliberate design choice). A community adapter exists at [pi-mcp-adapter](https://github.com/nicobailon/pi-mcp-adapter) but is unmaintained. Skein does not auto-detect Pi.

### Windows notes

`skein up` on Windows registers a Scheduled Task named `Skein\Daemon` at the
current user's logon — no admin elevation needed. The XML definition pins
`RestartOnFailure` to the same 3-retry/1-minute interval the launchd
`KeepAlive` and systemd `Restart=always` paths use, so reboot persistence
and crash recovery match the POSIX backends. State lives under
`%APPDATA%\skein\` (the same set of files macOS/Linux keep in
`~/.config/skein/`). To see or remove the task by hand:

```powershell
schtasks /Query  /TN "Skein\Daemon" /V
schtasks /Delete /TN "Skein\Daemon" /F   # equivalent to `skein down`
```

If `schtasks.exe` is missing (Server Core, nano-server, stripped containers)
the daemon falls back to a `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`
nohup-style spawn — survives terminal close but not reboot. Run `skein up`
again on first login in that case.

`skein up` is **idempotent** — safe to run repeatedly. It does:

| | |
|---|---|
| 1. Init | Generate bearer token + config (`~/.config/skein/config.json`) on first run |
| 2. Daemon | Install + start a background service (auto-relocates the venv to `~/.skein/venv` if needed for macOS TCC) |
| 3. Scope | Auto-detect from git remote (`git@github.com:user/repo.git` → `project:repo`) or cwd name |
| 4. Hooks | Drop `.claude/settings.json` + `.cursor/rules/skein.mdc` + `.skein/scope` so every LLM auto-recalls and auto-remembers |
| 5. Sync | Write MCP configs for every detected client (Claude Code, Cursor, Windsurf, Kiro, Codex, Hermes, Goose, Crush, gptme, Continue.dev, Antigravity, VS Code / Copilot, opencode) + AGENTS.md + CLAUDE.md |
| 6. Ingest | Index the codebase for semantic search — incremental, free re-runs |
| 7. Watcher | Spawn a session-scoped subprocess that re-ingests changed files within ~2 seconds. Survives terminal close; dies on logout (re-spawned by next `skein up`) |

To turn it off:

```bash
skein down       # stop the daemon and remove hooks from this project
skein restart    # restart the daemon
skein status     # see what's running (daemon + clients + counts)
skein doctor     # deeper diagnostic (logs, value distribution, inbox depth)
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

`skein up` (which also runs on `skein connect`) turns Skein from "tools the LLM can call" into "context that flows automatically without anyone asking." It writes a small set of files into the current project:

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
# Run once per project — installs hooks for every detected client:
cd ~/Documents/your-project
skein up

# What's wired up?
skein status

# Disconnect a client (removes the hooks):
skein connect cursor --remove
skein connect --all --remove   # disconnect everything
```

`skein up` writes the hooks into `.claude/settings.json` automatically (the per-project `.skein/scope` file pins which scope to use).

---

## Codebase RAG

Fragments are great for typed context — decisions, requirements, observations. They're not great for "find the function that does X across this 50k-LOC codebase." For that, Skein has a separate **chunks** layer.

### Ingest a codebase

`skein up` runs the initial ingest and registers a file-watcher that
keeps the chunk index current automatically. The visible knobs:

```bash
# First-time setup: indexes the cwd and starts the watcher.
skein up

# Force a full re-index (replaces the old `skein ingest .`):
skein doctor --reingest

# Re-embed every fragment under the active embedding provider
# (e.g. after switching from gemini to fastembed):
skein doctor --reindex-embeddings
```

What gets ingested:

- **Languages auto-detected** by extension: Python, JS, TS, Go, Rust, Java, Kotlin, Swift, Ruby, PHP, C, C++, C#, Scala, Clojure, Elixir, Haskell, Lua, Shell, SQL, Markdown, YAML, JSON, TOML, HTML, CSS, Dockerfile, Terraform, Protobuf, GraphQL, Vue, Svelte, Dart, R, …
- **Excluded by default**: `.git`, `node_modules`, `__pycache__`, `venv`, `.venv`, `dist`, `build`, `target`, `.next`, `.cache`, `_archive_v2`, …
- **Skipped**: files larger than 512 KB (override with `--max-bytes`), binary files, anything that fails UTF-8 decode

Each file is split into overlapping line windows (default 80 lines, 10-line overlap). Each chunk is hashed with SHA-256 — re-ingesting a file whose contents haven't changed is **free** (no DB write, no re-embedding API call).

### Search the ingested code

The agent is the canonical caller — the MCP `search_code` tool is what
Claude Code, Cursor, Codex, etc. invoke directly. Humans see the same
ranking through the REST endpoint:

```bash
# MCP — Claude Code, Cursor, etc. call this directly
search_code(query="how does auth work", scope="project:myapp")

# REST (same hybrid pipeline)
curl -X POST http://127.0.0.1:8765/v1/chunks/search \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query":"auth bearer","scope":"project:myapp","limit":5}'
```

Results return file path + line range + the matched chunk content, ranked by hybrid BM25 + vector + RRF.

### Stats / inspection

Per-scope chunk counts, language breakdown, and root summaries appear
in `skein doctor` (alongside fragment counts, value distribution, and
inbox depth). One-off cleanup is `skein doctor --clean`.

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
| `embedding_provider` | `fastembed` | `SKEIN_EMBEDDING_PROVIDER` |
| `bearer_token` | (generated) | `SKEIN_BEARER_TOKEN` |
| `default_scope` | `project:default` | `SKEIN_DEFAULT_SCOPE` |

Embedding providers: `fastembed` (local BAAI/bge-small-en-v1.5, 384-dim, default — no API key, ~130 MB one-time model download), `openai` (cloud, requires `OPENAI_API_KEY`), `bm25` (FTS5-only, no vector ranking), `hash` (tests-only).

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

Scopes are inferred automatically — `skein up` picks one from
`.skein/scope`, the git remote, or the directory name. Sub-scopes
(`org:`, `team:`) are created by the daemon the first time a fragment
references them; you never run a `scope create` command.

---

## Embedding quality

The default `fastembed` provider runs `BAAI/bge-small-en-v1.5` locally (384-dim, ONNX). First daemon startup downloads ~130 MB of model weights to `~/.cache/fastembed/`; after that every embedding is a sub-15 ms CPU call with no API key and no rate limits. Recall is hybrid: BM25 (FTS5) fused with vector cosine via Reciprocal Rank Fusion.

If you want OpenAI's `text-embedding-3-small` instead:

```bash
export OPENAI_API_KEY=your-key
pip install 'skn[openai]'
skein config set embedding_provider openai
```

If you can't or don't want to install fastembed, fall back to keyword-only:

```bash
skein config set embedding_provider bm25
```

> The previous `gemini` embedding provider was removed in iter 27 — its rate limits wedged the daemon's event loop. Existing configs naming `gemini` are silently aliased to `fastembed` on next load. **Gemini CLI** (the terminal agent) is being sunset by Google on June 18, 2026; **Antigravity** is the replacement and is already a fully-supported sync target. Skein continues to detect and connect `gemini_cli` for users still on it.

---

## License

MIT — see [LICENSE](LICENSE).
