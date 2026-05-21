"""Hand-rolled MCP Streamable HTTP server.

Implements the Model Context Protocol (MCP) specification using plain
FastAPI — no `mcp[cli]` SDK required (which needs Python 3.10+).

Protocol reference: https://spec.modelcontextprotocol.io/

Transport: Streamable HTTP (request/response only — no SSE in v1)
  POST /mcp        — JSON-RPC 2.0 request → JSON-RPC 2.0 response

Authentication: bearer token in the Authorization header, validated against
``cfg.bearer_token`` before any method dispatch. Same token as the REST API.

Supported JSON-RPC methods:
  initialize                — handshake
  tools/list                — advertise tools
  tools/call                — call a tool
  resources/list            — advertise resources
  resources/read            — read a resource
  prompts/list              — advertise prompts
  prompts/get               — get a prompt template

Tools (ADR-002 surface):
  recall(query, scope, types?, limit?)
  recall_one(fragment_id)
  remember(content, type, territory?, tags?, ttl_seconds?)
  note_decision(content, alternatives?, rationale?)
  claim_lease(glob, ttl_seconds, reason?)
  release_lease(lease_id)
  query_leases(scope?)

Resources:
  context://{scope}/state
  context://{scope}/decisions
  context://{scope}/agents-md
  context://{scope}/recent-commits

Prompts:
  session_start
"""
from __future__ import annotations

import json
import logging
import secrets
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("skein.mcp")

# MCP protocol version we advertise
MCP_PROTOCOL_VERSION = "2024-11-05"

router = APIRouter(tags=["mcp"])


# ---------------------------------------------------------------------------
# Auth — bearer token, validated at the transport layer.
# Without this guard any local process could call remember/recall/claim_lease.
# ---------------------------------------------------------------------------

def _check_mcp_auth(request: Request) -> Optional[JSONResponse]:
    """Return a 401/503 JSONResponse if the request is unauthenticated."""
    from .config import get_config
    cfg = get_config()
    if not cfg.bearer_token:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {
                "code": -32000,
                "message": "Skein not initialised. Run `skein init` first.",
            }},
            status_code=503,
        )
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):].strip()
    if not token or not secrets.compare_digest(token, cfg.bearer_token):
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {
                "code": -32001,
                "message": "Unauthorized: missing or invalid bearer token",
            }},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


# ---------------------------------------------------------------------------
# Entry point — POST /mcp
# ---------------------------------------------------------------------------

@router.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    """Handle a single JSON-RPC 2.0 request or a batch."""
    auth_error = _check_mcp_auth(request)
    if auth_error is not None:
        return auth_error
    try:
        body = await request.json()
    except Exception:
        return _error_response(None, -32700, "Parse error")

    # Batch requests
    if isinstance(body, list):
        responses = [await _handle_one(req, request) for req in body]
        return JSONResponse(responses)

    # Single request / notification
    if isinstance(body, dict):
        result = await _handle_one(body, request)
        if result is None:
            # Notification — no response per spec
            return JSONResponse(None, status_code=202)
        return JSONResponse(result)

    return _error_response(None, -32600, "Invalid Request")


# ---------------------------------------------------------------------------
# Method dispatcher
# ---------------------------------------------------------------------------

async def _handle_one(msg: dict[str, Any], request: Request) -> Optional[dict]:
    req_id = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params") or {}

    # Notification (no id) — process but don't respond
    is_notification = "id" not in msg

    try:
        result = await _dispatch(method, params, request)
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except McpError as e:
        if is_notification:
            return None
        return _error_response(req_id, e.code, e.message, e.data)
    except Exception as e:
        logger.exception("Unexpected error in MCP method %s", method)
        if is_notification:
            return None
        return _error_response(req_id, -32603, "Internal error", {"detail": str(e)})


async def _dispatch(method: str, params: dict[str, Any], request: Request) -> Any:
    from .dependencies import get_provider, get_storage

    storage = get_storage()
    provider = get_provider()

    # ---- Lifecycle ----
    if method == "initialize":
        # Capture which client is calling, keyed by bearer-token prefix. Lets
        # every subsequent tool call attribute its writes to the originating
        # tool without the user managing per-client tokens.
        _remember_initiating_client(params, request, storage)
        return _handle_initialize(params, storage)

    if method == "notifications/initialized":
        return {}  # ack

    if method == "ping":
        return {}

    # ---- Tools ----
    if method == "tools/list":
        return {"tools": _TOOLS}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        return await _call_tool(name, args, storage, provider, request)

    # ---- Resources ----
    if method == "resources/list":
        # Static resources only — per spec the `{scope}` URIs are templates
        # and belong under resources/templates/list, not here.
        return {"resources": []}

    if method == "resources/templates/list":
        return {"resourceTemplates": _RESOURCE_TEMPLATES}

    if method == "resources/read":
        uri = params.get("uri", "")
        return await _read_resource(uri, storage, request)

    # ---- Prompts ----
    if method == "prompts/list":
        return {"prompts": _PROMPTS}

    if method == "prompts/get":
        import asyncio
        name = params.get("name")
        args = params.get("arguments") or {}
        # _get_prompt runs a sync recall (with embedding) — offload to
        # a worker thread to keep the asyncio event loop responsive.
        return await asyncio.to_thread(_get_prompt, name, args, storage)

    raise McpError(-32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

def _normalize_client_name(raw: str) -> str:
    """Normalize a clientInfo.name into a canonical lowercase-hyphenated form.

    Examples: ``"Claude Code"`` → ``"claude-code"``, ``"Cursor"`` → ``"cursor"``,
    ``"Gemini CLI"`` → ``"gemini-cli"``, ``""`` → ``"unknown"``.
    """
    if not raw or not isinstance(raw, str):
        return "unknown"
    s = raw.strip().lower()
    # Collapse runs of whitespace/dots/underscores into single hyphens
    out_chars = []
    last_dash = False
    for ch in s:
        if ch.isalnum():
            out_chars.append(ch)
            last_dash = False
        elif not last_dash:
            out_chars.append("-")
            last_dash = True
    result = "".join(out_chars).strip("-")
    return result or "unknown"


def _remember_initiating_client(params: dict[str, Any], request: Request, storage: Any) -> None:
    """Record the (token_prefix, client_name) pairing for this connection.

    Reads ``params.clientInfo.name`` per MCP spec; falls back to ``"unknown"``
    if the client didn't introduce itself.
    """
    try:
        from .auth import token_prefix as _prefix
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip() if auth_header else ""
        if not token:
            return
        prefix = _prefix(token)
        client_info = params.get("clientInfo") or {}
        name = _normalize_client_name(client_info.get("name", ""))
        display = client_info.get("name") if client_info else None
        storage.upsert_mcp_client(prefix, name, display_name=display)
    except Exception:
        # Never let a logging-style helper break the initialize handshake.
        logger.debug("failed to record client for token", exc_info=True)


def _client_name_for_request(request: Request, storage: Any) -> str:
    """Best-effort lookup of which LLM client this request came from."""
    try:
        from .auth import token_prefix as _prefix
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip() if auth_header else ""
        if not token:
            return "unknown"
        prefix = _prefix(token)
        return storage.get_client_for_token_prefix(prefix) or "unknown"
    except Exception:
        return "unknown"


def _handle_initialize(params: dict, storage: Any) -> dict:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": {"name": "skein", "version": "0.1.0"},
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"listChanged": False, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        # Per MCP spec, `instructions` is server-supplied text the client
        # SHOULD include in its system prompt. We use it to auto-inject the
        # recall-first guidance — avoids the AI consumer needing to be told
        # by the user/AGENTS.md to call `recall` first on every turn.
        # Iter 29 day-one: the text is now dynamic — appends a "this project"
        # block with fragment count + last-24h cross-LLM activity. Empty
        # stores get a starter prompt instead of a passive welcome.
        "instructions": _build_initialize_instructions(storage),
    }


def _build_initialize_instructions(storage: Any) -> str:
    """Render the dynamic onboarding greeting that ships in the MCP
    ``initialize.instructions`` field on every connection.

    Three blocks: the static recall-first rules, then per-DB state
    (fragment count + last-24h cross-LLM activity), then a one-line
    "what to try first" tailored to whether the store is empty or full.
    """
    try:
        stats = storage.stats()
        fragment_count = int(stats.get("fragments", 0))
    except Exception:
        fragment_count = 0
    try:
        activity = storage.recent_writes_by_tool(hours=24)
    except Exception:
        activity = {}

    lines: list[str] = [_RECALL_FIRST_TEXT.rstrip(), ""]

    if fragment_count == 0:
        # Empty store — fresh install / never-used DB. Don't be cheerful;
        # be specific about what's coming and what the LLM can do now.
        lines.extend([
            "Project state: no fragments stored yet. Cold-start ingest is "
            "queued — recent git commits, README claims, and dep manifests "
            "will appear in `recall` within ~10s. In the meantime, every "
            "`remember` / `note_decision` call you make will be the first "
            "thing future sessions see.",
        ])
    else:
        lines.append(f"Project state: {fragment_count} fragments stored.")
        if activity:
            # Limit to top 4 tools so the system prompt stays tight.
            top = sorted(activity.items(), key=lambda x: -x[1])[:4]
            parts = [f"{tool} ({count})" for tool, count in top]
            lines.append(
                f"Cross-tool activity (last 24h): {', '.join(parts)}."
            )
        else:
            lines.append(
                "No writes in the last 24 h — Skein is quiet. Connecting "
                "another LLM (`skein connect`) compounds the value: every "
                "decision you record here surfaces in `cursor` / `codex` "
                "sessions on the same project."
            )

    lines.extend([
        "",
        "Quick start: call `project_briefing` for the dashboard, or "
        "`recall(\"<your task>\")` to load relevant context.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "project_briefing",
        "description": (
            "Returns the project's current state in ONE call — fragment counts "
            "by type, recent decisions, daemon health, recommended next "
            "action. Use this BEFORE reading any source file or calling "
            "multiple `recall`s when you need a project overview. Returns in "
            "<50ms, costs ~300 tokens. The fastest path to 'what is "
            "happening here?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": (
                        "Scope handle, e.g. 'project:myapp'. "
                        "Omit to auto-detect from cwd."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "recall",
        "description": (
            "Returns ranked context fragments (decisions, facts, observations, "
            "preferences, state, requirements) matching a natural-language query. "
            "Use BEFORE reading source files when you need project history, prior "
            "decisions, or non-obvious 'why' / 'how' context. "
            "Returns top-K in <100ms, ~30 tokens per fragment — one `recall` "
            "typically replaces 5+ `read_file` calls. Each result carries a "
            "`quality` bucket (high/medium/low/none) derived from the underlying "
            "cosine similarity; if the top result is `quality=none`, Skein has "
            "no high-signal context for that query and you should fall back to "
            "source. Scope auto-detected from cwd."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "scope": {"type": "string", "description": "Scope handle, e.g. 'project:myapp'. Omit to auto-detect."},
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by fragment type(s): preference, fact, decision, state, observation, requirement, procedure, conversation",
                },
                "territory": {"type": "string", "description": "Filter by territory prefix, e.g. 'backend/auth'"},
                "limit": {"type": "integer", "default": 10, "description": "Max results (1–50)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_one",
        "description": (
            "Retrieve the full content of a specific fragment by ID. "
            "Use after `recall` when you saw a truncated fragment and need the "
            "complete text. <10ms, ~150 tokens average."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fragment_id": {"type": "string"},
            },
            "required": ["fragment_id"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Persists a fact, observation, decision, preference, state, or "
            "requirement to the project's context bus. Call AFTER making a "
            "non-obvious decision so other tools (you in a later session, or "
            "other LLMs) can recall it. Dedupes by content+scope+source_tool — "
            "safe to call eagerly. Returns the fragment ID. ~5ms. "
            "Scope auto-detected from cwd."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The context to store"},
                "type": {
                    "type": "string",
                    "enum": ["preference", "fact", "decision", "state",
                             "observation", "requirement", "procedure", "conversation"],
                    "description": "Fragment type",
                },
                "scope": {"type": "string", "description": "Scope handle. Omit to auto-detect."},
                "territory": {"type": "string", "description": "File/domain area, e.g. 'backend/auth'"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "ttl_seconds": {"type": "integer", "description": "TTL override. 0 = permanent."},
            },
            "required": ["content", "type"],
        },
    },
    {
        "name": "note_decision",
        "description": (
            "Persists an architectural or technical decision with structured "
            "`alternatives` and `rationale` fields. Call AFTER picking an "
            "approach when the WHY matters (architectural choices, library "
            "picks, performance trade-offs). Higher signal than "
            "`remember(type='decision')` because the structured fields preserve "
            "the reasoning across tools and sessions. ~5ms. Scope auto-detected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The decision"},
                "scope": {"type": "string", "description": "Scope handle. Omit to auto-detect."},
                "territory": {"type": "string"},
                "alternatives": {"type": "string", "description": "What alternatives were considered"},
                "rationale": {"type": "string", "description": "Why this decision was made"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"],
        },
    },
    {
        "name": "claim_lease",
        "description": (
            "Reserves a file-glob area for exclusive editing by your session. "
            "Call BEFORE multi-step refactors to prevent parallel agents from "
            "stepping on each other (e.g. two Claude Code sessions both "
            "rewriting `auth/`). Other agents see LEASE_CONFLICT if they try "
            "the same glob. Defaults to 5-min TTL. ~5ms."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "glob": {"type": "string", "description": "File glob, e.g. 'backend/auth/**'"},
                "scope": {"type": "string", "description": "Scope handle. Omit to auto-detect."},
                "ttl_seconds": {"type": "integer", "default": 300},
                "reason": {"type": "string"},
            },
            "required": ["glob"],
        },
    },
    {
        "name": "release_lease",
        "description": (
            "Release a previously acquired lease before its TTL expires. "
            "Call when you're done editing the area you claimed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lease_id": {"type": "string"},
            },
            "required": ["lease_id"],
        },
    },
    {
        "name": "query_leases",
        "description": (
            "List active leases for a scope. Use as a coordination check before "
            "claiming your own lease — discovers what other agents are currently "
            "editing. ~5ms."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "supersede",
        "description": (
            "Atomically replaces an outdated fragment with new content. Marks "
            "the old fragment stale (reason: 'superseded by <new_id>') and "
            "creates a new fragment inheriting the old one's scope/type/"
            "territory/tags. Use when a prior decision/fact is no longer correct "
            "(e.g. 'use Redis' → 'use Memcached'). recall() then prefers the "
            "new fragment AND surfaces the supersede chain — uniquely Skein's "
            "decision archaeology. ~10ms."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "old_fragment_id": {"type": "string", "description": "ID of the fragment to supersede"},
                "new_content": {"type": "string", "description": "Replacement content"},
                "reason": {"type": "string", "description": "Optional note explaining why the old fragment is outdated"},
                "type": {
                    "type": "string",
                    "enum": ["preference", "fact", "decision", "state",
                             "observation", "requirement", "procedure", "conversation"],
                    "description": "Override the type — defaults to the old fragment's type.",
                },
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Override tags — defaults to the old fragment's tags."},
            },
            "required": ["old_fragment_id", "new_content"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Hybrid (BM25 + semantic) search over the project's indexed code "
            "and docs. Returns ranked snippets with file paths and line ranges. "
            "Use when you need to FIND something by meaning ('the place that "
            "handles rate limiting') rather than read a file whose path you "
            "already know. Faster and more relevant than `grep` for conceptual "
            "queries; complements `read_file` for known paths. <50ms, ~100 "
            "tokens per snippet. Code must be ingested via `skein ingest` first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "scope": {"type": "string", "description": "Scope handle. Omit to auto-detect."},
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by language: python, typescript, go, etc.",
                },
                "source_root": {
                    "type": "string",
                    "description": "Restrict to a specific ingest root (e.g. project name).",
                },
                "limit": {"type": "integer", "default": 8, "description": "Max results (1–50)"},
            },
            "required": ["query"],
        },
    },
    # ADR-002 / iter 26 — agent-facing controls for the fragment-value
    # system (Q-05). The user never types these; the agent invokes them
    # from natural language like "remember this is critical" or "this
    # decision turned out wrong, ignore it." No CLI surface.
    {
        "name": "boost",
        "description": (
            "Pin a fragment to a high recall-time value. Use when the user "
            "says \"this is important\" / \"always remember this\" / \"keep "
            "this in mind\" about a specific fragment. Survives the daemon's "
            "decay loop so the boost sticks across sessions. Takes a "
            "fragment id (full or 8-char prefix). Returns the new value. ~5ms."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fragment_id": {"type": "string", "description": "Fragment id or 8-char prefix."},
                "value": {
                    "type": "number",
                    "description": "Target value in [0.05, 1.0]. Default 1.0.",
                    "default": 1.0,
                },
            },
            "required": ["fragment_id"],
        },
    },
    {
        "name": "bury",
        "description": (
            "Drop a fragment's recall-time value to the floor. Use when the "
            "user says \"this is wrong\" / \"forget this\" / \"ignore this "
            "fragment.\" Doesn't delete the fragment — it stays in the audit "
            "log — but it's effectively hidden from default recall. Takes a "
            "fragment id (full or 8-char prefix). ~5ms."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fragment_id": {"type": "string", "description": "Fragment id or 8-char prefix."},
            },
            "required": ["fragment_id"],
        },
    },
    {
        "name": "archaeology",
        "description": (
            "Reconstruct the provenance of a decision: who created the "
            "fragment, in which session, against which commit, what it "
            "superseded, what superseded it. Use when answering \"why did "
            "we decide X?\" — Skein has the full origin story even if the "
            "commit message doesn't. Returns a structured trace. ~50ms."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Fragment id, 8-char prefix, or natural-language search.",
                },
                "scope": {"type": "string", "description": "Scope handle. Omit to auto-detect."},
                "limit": {"type": "integer", "default": 5, "description": "Max traces (1–20)"},
            },
            "required": ["query"],
        },
    },
]


_GIT_HEAD_CACHE: dict[str, tuple] = {}  # cwd → (commit_hash, expires_at)


def _resolve_git_head() -> Optional[str]:
    """Cheap, cached lookup of ``git rev-parse HEAD`` for the daemon's cwd.

    Cached for 30s — fast enough for high-frequency MCP calls without going
    stale during long sessions. Returns None on any error (no git, no repo,
    binary not found).
    """
    import os
    import subprocess
    import time as _time
    cwd = os.getcwd()
    entry = _GIT_HEAD_CACHE.get(cwd)
    if entry and entry[1] > _time.time():
        return entry[0]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=1.0,
        )
        sha = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        sha = None
    _GIT_HEAD_CACHE[cwd] = (sha, _time.time() + 30.0)
    return sha


def _auto_scope() -> str:
    """Resolve scope using the same precedence as the CLI / hooks.

    Falls back to ``personal:scratch`` if nothing matches — keeps the MCP
    tools usable from any cwd without forcing the AI to invent a scope handle.
    """
    from .config import get_config
    from .scope_resolver import resolve_scope
    try:
        cfg = get_config()
        handle, _ = resolve_scope(None, config_default=cfg.default_scope)
        return handle
    except Exception:
        return "personal:scratch"


# ---------------------------------------------------------------------------
# project_briefing — single-call project snapshot
# ---------------------------------------------------------------------------

# Fragment types we always surface in the briefing, even when the count is 0.
# Keeps the response shape stable so LLMs can rely on `decision`/`fact`/…
# always being present.
_BRIEFING_TYPES = ("decision", "fact", "observation", "preference",
                   "state", "requirement", "procedure", "conversation")


def build_briefing(storage: Any, scope_handle: str) -> dict[str, Any]:
    """Pure builder for the project-briefing payload.

    Kept transport-agnostic so the MCP handler, REST router, and tests can all
    call it without going through JSON-RPC. Returns the dict the spec
    describes; callers serialise it (JSON for HTTP, MCP text content, etc.).
    """
    from .config import get_config
    from .server import get_daemon_uptime_seconds

    cfg = get_config()
    scope = storage.get_scope(scope_handle)

    if scope is None:
        # Permissive: an LLM may call briefing on a brand-new project before
        # any fragments exist. Return zeros rather than 404 — the MCP tool
        # should be safe to call from any cwd.
        type_counts: dict[str, int] = {}
        recent_decisions: list[dict[str, Any]] = []
        fragment_total = 0
    else:
        type_counts = storage.count_fragments_by_type(scope.id)
        fragment_total = storage.count_fragments_in_scope(scope.id)
        recent_frags = storage.list_fragments(
            scope_id=scope.id, type_filter="decision",
            include_stale=False, limit=5,
        )
        recent_decisions = []
        for f in recent_frags:
            first_line = (f.content or "").splitlines()[0] if f.content else ""
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            recent_decisions.append({
                "id_short": f.id[:8],
                "content_first_line": first_line,
                "created_by_tool": f.created_by_tool,
                "created_at": f.created_at,
                "tags": list(f.tags),
            })

    # Pad the counts dict so every known type appears, even with 0.
    fragment_counts = {t: int(type_counts.get(t, 0)) for t in _BRIEFING_TYPES}
    # Surface any unexpected types too (forward-compat for new enum values).
    for t, c in type_counts.items():
        fragment_counts.setdefault(t, int(c))

    chunks_total = int(storage.count_chunks())
    active_inbox_count = int(storage.count_extraction_candidates(status="pending"))

    # Heuristic recommendation — short, LLM-readable.
    decisions_count = fragment_counts.get("decision", 0)
    if active_inbox_count > 0:
        next_action = (
            f"Review {active_inbox_count} pending fragments via skein inbox"
        )
    elif decisions_count < 10:
        next_action = (
            "Few decisions captured; this project is still bootstrapping memory"
        )
    else:
        next_action = "Project is healthy; use recall<query> for specific context"

    return {
        "scope": scope_handle,
        "fragment_counts": fragment_counts,
        "fragment_total": int(fragment_total),
        "chunks_total": chunks_total,
        "recent_decisions": recent_decisions,
        "active_inbox_count": active_inbox_count,
        "embedding_provider": cfg.embedding_provider,
        "daemon": {
            "version": "0.1.0",
            "uptime_seconds": get_daemon_uptime_seconds(),
            "db_path": storage.db_path,
        },
        "next_recommended_action": next_action,
    }


# ---------------------------------------------------------------------------
# Tool call handler
# ---------------------------------------------------------------------------

async def _call_tool(
    name: str,
    args: dict[str, Any],
    storage: Any,
    provider: Any,
    request: Request,
) -> dict:
    import asyncio
    from .auth import token_prefix
    from .config import get_config
    from .models import (
        FragmentCreate, LeaseCreate, RecallRequest, IdentityCreate,
    )
    from .retrieval import recall as do_recall
    from .storage import ConflictError

    cfg = get_config()

    # Resolve caller identity from request auth header
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip() if auth_header else ""
    handle = f"user:{token_prefix(token)}" if token else "user:mcp"
    identity = storage.get_or_create_identity(IdentityCreate(
        handle=handle, type="llm", name="mcp-caller",
    ))
    owner_id = identity.id

    # Auto-resolve scope if the caller omitted it. Saves the AI from having
    # to pass `scope="project:foo"` on every recall — match the same
    # precedence as the CLI (.skein/scope > SKEIN_SCOPE > config default).
    args.setdefault("scope", _auto_scope())

    # Provenance (iter 14.0): which LLM client is calling? Looked up by
    # token-prefix, populated in `initialize`. Tools may optionally pass
    # `session_id` and `files_open` in their args for richer provenance.
    client_name = _client_name_for_request(request, storage)
    session_id = args.pop("session_id", None) if "session_id" in args else None
    files_open = args.pop("files_open", None) if "files_open" in args else None
    if not isinstance(files_open, list):
        files_open = []
    git_head = _resolve_git_head()

    def _ensure_scope(scope_handle: str):
        """Get-or-create the scope so first-time agents don't fail."""
        s = storage.get_scope(scope_handle)
        if s:
            return s
        from .models import ScopeCreate
        prefix = scope_handle.split(":", 1)[0]
        stype = prefix if prefix in {"public", "org", "team", "project", "personal"} else "project"
        return storage.create_scope(ScopeCreate(
            handle=scope_handle, type=stype,
            name=scope_handle.split(":", 1)[-1], owner_id=owner_id,
        ))

    def _resolve_fragment_id(storage_, prefix: str) -> Optional[str]:
        """Accept a full UUID or an 8+ char prefix and return the full id.

        Used by boost/bury/archaeology so the agent can paste the short id
        the recall output already shows. Returns None on miss / ambiguous.
        """
        if not prefix:
            return None
        # Exact id?
        if len(prefix) == 36 and "-" in prefix:
            row = storage_._conn.execute(
                "SELECT id FROM fragments WHERE id = ?", (prefix,),
            ).fetchone()
            return row["id"] if row else None
        rows = storage_._conn.execute(
            "SELECT id FROM fragments WHERE id LIKE ? LIMIT 2",
            (prefix + "%",),
        ).fetchall()
        if len(rows) != 1:
            return None
        return rows[0]["id"]

    def _resolve_scope_from_cwd_or_default() -> str:
        """Pick a sensible default scope when the caller didn't pass one.

        Mirrors what `recall` / `remember` do today via _project_scope —
        keeps archaeology consistent with the rest of the surface.
        """
        try:
            from .scope_resolver import resolve_scope
            from .config import get_config
            return resolve_scope(None, config_default=get_config().default_scope)[0]
        except Exception:
            return "project:default"

    # ---- project_briefing ----
    if name == "project_briefing":
        briefing = build_briefing(storage, args["scope"])
        return _tool_text(json.dumps(briefing, indent=2))

    # ---- recall ----
    if name == "recall":
        from .events import log_event
        req = RecallRequest(
            query=args["query"],
            scope=args["scope"],
            types=args.get("types"),
            territory=args.get("territory"),
            limit=args.get("limit", 10),
        )
        # Offload to a worker thread — do_recall's embedding call + SQLite
        # queries are synchronous and would otherwise block the asyncio
        # event loop (and /health) during fastembed's first-call warm-up
        # or any slow embed/search.
        response = await asyncio.to_thread(do_recall, req, storage, provider)
        log_event("recall", scope=args["scope"], query=args["query"][:120], hits=response.total)
        # Iter 29 day-one: empty-OR-low-quality recall offers a write
        # suggestion. With hybrid BM25+vector+RRF, true `[]` results are
        # rare — but a top-quality of "none" means Skein has nothing
        # high-signal for the query. Both cases deserve the same nudge.
        is_low_signal = (
            not response.results
            or response.results[0].quality == "none"
        )
        if is_low_signal:
            query = args.get("query", "")
            # Filters: real-question shape (≥10 chars + whitespace). Skips
            # one-word "test" / "foo" probes that don't make good writes.
            if len(query) >= 10 and " " in query.strip():
                escaped = query.replace('"', '\\"')[:120]
                # Iter 30 wording fix: "Found 0 fragments" reads as "the
                # store has 0 fragments total" — misleading users into
                # thinking Skein is empty when actually the store has
                # plenty and just nothing matches this specific query.
                # Always state the total so the failure mode is unambiguous.
                try:
                    total_in_scope = storage.count_fragments_in_scope(
                        response.scope if hasattr(response, "scope") else args["scope"],
                    )
                except Exception:
                    total_in_scope = None
                total_note = (
                    f" (Skein has {total_in_scope} fragments in this scope, "
                    "none of them semantically related to your query.)"
                    if total_in_scope is not None and total_in_scope > 0
                    else ""
                )
                if not response.results:
                    preface = (
                        f"No fragment in Skein matched {query!r}.{total_note} "
                    )
                else:
                    preface = (
                        f"No high-signal match for {query!r} — the top of "
                        f"{response.total} candidate fragments scored as "
                        f"low-signal (quality=none).{total_note} "
                    )
                suggestion = (
                    f"{preface}"
                    f"If you have context for this, call "
                    f"`remember(content=\"<your answer or decision>\", "
                    f"type=\"fact\", scope=\"{args['scope']}\")` so the "
                    f"next session (or another LLM working on this project) "
                    f"sees it. Suggested writeup query: \"{escaped}\"."
                )
                return _tool_text(suggestion)
            if not response.results:
                return _tool_text("No relevant context found.")
            # Low-signal but unusable query — fall through to normal rendering.
        # iter 24: lead with the quality bucket — it's the only signal callers
        # can route on without knowing what RRF/BM25/cosine look like.
        top_quality = response.results[0].quality
        header = f"Found {response.total} fragments for query: {response.query!r}"
        if top_quality == "none":
            header += (
                "\n[top match is low-signal — Skein lacks high-quality context "
                "for this query; fall back to source.]"
            )
        elif top_quality == "low":
            header += "\n[top match is low quality — verify before relying.]"
        lines = [header + "\n"]
        for r in response.results:
            f = r.fragment
            territory_note = f" [{f.territory}]" if f.territory else ""
            tags_note = f" #{' #'.join(f.tags)}" if f.tags else ""
            cos_note = f" cos={r.cosine:.2f}" if r.cosine is not None else ""
            lines.append(
                f"[{r.rank}] {f.type.upper()}{territory_note}{tags_note} "
                f"(quality={r.quality}{cos_note}, id={f.id[:8]}…)\n"
                f"  {f.content}\n"
            )
        return _tool_text("\n".join(lines))

    # ---- recall_one ----
    if name == "recall_one":
        frag = storage.get_fragment(args["fragment_id"])
        if not frag:
            return _tool_text("Fragment not found.")
        return _tool_text(
            f"Fragment {frag.id}\n"
            f"Type: {frag.type}  Territory: {frag.territory or '—'}\n"
            f"Tags: {', '.join(frag.tags) or '—'}\n"
            f"Created: {frag.created_at}\n\n"
            f"{frag.content}"
        )

    # ---- remember ----
    if name == "remember":
        scope = _ensure_scope(args["scope"])
        data = FragmentCreate(
            content=args["content"],
            type=args["type"],
            scope_id=scope.id,
            owner_id=owner_id,
            territory=args.get("territory"),
            tags=args.get("tags", []),
            ttl_seconds=args.get("ttl_seconds"),
            created_by_tool=client_name,
            created_in_session_id=session_id,
            created_against_commit=git_head,
            files_open_at_creation=files_open,
            extraction_method="explicit",
            extraction_confidence=1.0,
        )

        from .embeddings import vec_to_bytes
        from .models import CommitCreate
        embedding_bytes = None
        try:
            vec = await asyncio.to_thread(provider.embed_one, args["content"])
            embedding_bytes = vec_to_bytes(vec)
        except Exception:
            pass

        commit = storage.create_commit(CommitCreate(
            author_id=owner_id, scope_id=scope.id,
            message=f"[mcp] add {args['type']}: {args['content'][:60]}",
        ))
        frag = storage.create_fragment(data, commit_id=commit.id, embedding=embedding_bytes)
        storage._conn.execute(
            "UPDATE commits SET fragments_added = ? WHERE id = ?",
            (f'["{frag.id}"]', commit.id),
        )
        from .events import log_event
        log_event(
            "remember", scope=args["scope"], fragment_id=frag.id,
            type=frag.type, preview=args["content"][:80],
        )
        return _tool_text(f"Stored fragment {frag.id[:8]}… (type={frag.type})")

    # ---- note_decision ----
    if name == "note_decision":
        parts = [args["content"]]
        if args.get("alternatives"):
            parts.append(f"\nAlternatives considered: {args['alternatives']}")
        if args.get("rationale"):
            parts.append(f"\nRationale: {args['rationale']}")
        full_content = "".join(parts)

        scope = _ensure_scope(args["scope"])

        from .embeddings import vec_to_bytes
        from .models import CommitCreate
        embedding_bytes = None
        try:
            vec = await asyncio.to_thread(provider.embed_one, full_content)
            embedding_bytes = vec_to_bytes(vec)
        except Exception:
            pass

        commit = storage.create_commit(CommitCreate(
            author_id=owner_id, scope_id=scope.id,
            message=f"[mcp] note_decision: {args['content'][:60]}",
        ))
        data = FragmentCreate(
            content=full_content, type="decision",
            scope_id=scope.id, owner_id=owner_id,
            territory=args.get("territory"),
            tags=args.get("tags", []),
            created_by_tool=client_name,
            created_in_session_id=session_id,
            created_against_commit=git_head,
            files_open_at_creation=files_open,
            extraction_method="explicit",
            extraction_confidence=1.0,
        )
        frag = storage.create_fragment(data, commit_id=commit.id, embedding=embedding_bytes)
        storage._conn.execute(
            "UPDATE commits SET fragments_added = ? WHERE id = ?",
            (f'["{frag.id}"]', commit.id),
        )
        from .events import log_event
        log_event(
            "note_decision", scope=args["scope"], fragment_id=frag.id,
            preview=args["content"][:80],
        )
        return _tool_text(f"Decision recorded: {frag.id[:8]}…")

    # ---- claim_lease ----
    if name == "claim_lease":
        scope = _ensure_scope(args["scope"])
        conflict = storage.check_lease_conflict(scope.id, args["glob"])
        if conflict and conflict.owner_id != owner_id:
            return _tool_text(
                f"LEASE_CONFLICT: '{args['glob']}' is held by {conflict.owner_id[:8]}… "
                f"(expires {conflict.expires_at}). "
                f"Wait for it to expire or use release_lease({conflict.id})."
            )
        data = LeaseCreate(
            scope_id=scope.id,
            glob=args["glob"],
            owner_id=owner_id,
            ttl_seconds=args.get("ttl_seconds", 300),
            reason=args.get("reason"),
        )
        lease = storage.acquire_lease(data)
        from .events import log_event
        log_event(
            "claim_lease", scope=args["scope"], lease_id=lease.id,
            glob=lease.glob, expires_at=lease.expires_at,
        )
        return _tool_text(
            f"Lease acquired: {lease.id[:8]}… on '{lease.glob}' "
            f"(expires {lease.expires_at})"
        )

    # ---- release_lease ----
    if name == "release_lease":
        released = storage.release_lease(args["lease_id"], owner_id)
        if released:
            from .events import log_event
            log_event("release_lease", lease_id=args["lease_id"])
            return _tool_text(f"Lease {args['lease_id'][:8]}… released.")
        return _tool_text("Lease not found or not owned by you.")

    # ---- supersede ----
    if name == "supersede":
        from .embeddings import vec_to_bytes
        from .models import CommitCreate, FragmentUpdate

        old_id = args["old_fragment_id"]
        old = storage.get_fragment(old_id)
        if not old:
            return _tool_text(f"Fragment {old_id} not found — nothing to supersede.")
        if old.is_stale:
            return _tool_text(
                f"Fragment {old_id[:8]}… is already stale "
                f"(reason: {old.stale_reason or 'unknown'}). "
                "Use `remember` to add a new fragment instead."
            )

        # Inherit scope/type/territory/tags unless overridden
        new_type = args.get("type") or old.type
        new_tags = args.get("tags") if args.get("tags") is not None else list(old.tags)
        new_content = args["new_content"]

        embedding_bytes = None
        try:
            vec = await asyncio.to_thread(provider.embed_one, new_content)
            embedding_bytes = vec_to_bytes(vec)
        except Exception:
            pass

        commit = storage.create_commit(CommitCreate(
            author_id=owner_id, scope_id=old.scope_id,
            message=f"[mcp] supersede {old_id[:8]}…: {new_content[:60]}",
        ))
        new_frag = storage.create_fragment(
            FragmentCreate(
                content=new_content, type=new_type,
                scope_id=old.scope_id, owner_id=owner_id,
                territory=old.territory, tags=new_tags,
                created_by_tool=client_name,
                created_in_session_id=session_id,
                created_against_commit=git_head,
                files_open_at_creation=files_open,
                supersedes_fragment_id=old.id,
                extraction_method="explicit",
                extraction_confidence=1.0,
            ),
            commit_id=commit.id, embedding=embedding_bytes,
        )

        # Now mark the old fragment stale, referencing the replacement.
        stale_reason = f"superseded by {new_frag.id}"
        if args.get("reason"):
            stale_reason += f" — {args['reason']}"
        try:
            storage.update_fragment(old_id, FragmentUpdate(
                is_stale=True,
                stale_reason=stale_reason,
                expected_version=old.version,
            ))
        except ConflictError:
            # Someone else modified the old fragment between our read and
            # write. The new fragment is already created — surface this so
            # the caller knows the old one wasn't actually marked stale.
            return _tool_text(
                f"New fragment {new_frag.id[:8]}… created, but the old "
                f"fragment {old_id[:8]}… was modified concurrently and "
                "could not be marked stale. Retry with `mark_stale` if needed."
            )

        storage._conn.execute(
            "UPDATE commits SET fragments_added = ? WHERE id = ?",
            (f'["{new_frag.id}"]', commit.id),
        )
        from .events import log_event
        log_event(
            "supersede", scope=args["scope"],
            old_fragment_id=old_id, new_fragment_id=new_frag.id,
            preview=new_content[:80],
        )
        return _tool_text(
            f"Superseded {old_id[:8]}… → {new_frag.id[:8]}… "
            f"(type={new_type})"
        )

    # ---- search_code ----
    if name == "search_code":
        from .models import ChunkSearchRequest
        from .retrieval import search_chunks

        # Ensure the scope exists so a fresh project can call search_code
        # without bouncing on "Scope not found" before any ingestion.
        _ensure_scope(args["scope"])
        req = ChunkSearchRequest(
            query=args["query"],
            scope=args["scope"],
            languages=args.get("languages"),
            source_root=args.get("source_root"),
            limit=args.get("limit", 8),
        )
        # Offloaded for the same reason as do_recall above — keeps the
        # event loop responsive to /health during fastembed warm-up.
        response = await asyncio.to_thread(search_chunks, req, storage, provider)
        if not response.results:
            return _tool_text(
                f"No code chunks found for {response.query!r}.\n"
                f"Has the codebase been ingested? Run `skein ingest <path>` first."
            )
        lines = [f"Found {response.total} code chunks for query: {response.query!r}\n"]
        for r in response.results:
            c = r.chunk
            sym = f" {c.symbol_name}" if c.symbol_name else ""
            lang = f" ({c.language})" if c.language else ""
            cos_note = f" cos={r.cosine:.2f}" if r.cosine is not None else ""
            lines.append(
                f"[{r.rank}] {c.source_path}:{c.line_start}-{c.line_end}{lang}{sym}  "
                f"quality={r.quality}{cos_note}\n"
                f"```{c.language or ''}\n{c.content}\n```\n"
            )
        return _tool_text("\n".join(lines))

    # ---- boost (ADR-002 / iter 26) ----
    if name == "boost":
        from .events import log_event
        from .models import FragmentUpdate
        frag_id_prefix = args["fragment_id"]
        target_value = float(args.get("value", 1.0))
        if not (0.05 <= target_value <= 1.0):
            return _tool_text(
                f"Error: value must be in [0.05, 1.0]; got {target_value}."
            )
        full_id = _resolve_fragment_id(storage, frag_id_prefix)
        if full_id is None:
            return _tool_text(
                f"No fragment matching prefix {frag_id_prefix!r}."
            )
        # Direct column write — value isn't part of the FragmentUpdate
        # OCC contract (it's a daemon-managed signal), so we go around the
        # ORM. Atomic single-statement UPDATE.
        n = storage._conn.execute(
            "UPDATE fragments SET value = ?, updated_at = datetime('now') WHERE id = ?",
            (target_value, full_id),
        ).rowcount
        storage._conn.commit()
        if n == 0:
            return _tool_text(f"Fragment {full_id[:8]}… not found.")
        log_event("boost", scope=None, fragment_id=full_id, value=target_value)
        return _tool_text(
            f"Boosted {full_id[:8]}… to value={target_value:.2f}. "
            f"Will outrank lower-value fragments on the same query."
        )

    # ---- bury (ADR-002 / iter 26) ----
    if name == "bury":
        from .events import log_event
        frag_id_prefix = args["fragment_id"]
        full_id = _resolve_fragment_id(storage, frag_id_prefix)
        if full_id is None:
            return _tool_text(
                f"No fragment matching prefix {frag_id_prefix!r}."
            )
        n = storage._conn.execute(
            "UPDATE fragments SET value = 0.05, updated_at = datetime('now') WHERE id = ?",
            (full_id,),
        ).rowcount
        storage._conn.commit()
        if n == 0:
            return _tool_text(f"Fragment {full_id[:8]}… not found.")
        log_event("bury", scope=None, fragment_id=full_id)
        return _tool_text(
            f"Buried {full_id[:8]}… to value=0.05. "
            f"Hidden from default recall; still in the audit log."
        )

    # ---- archaeology (ADR-002 / iter 26) ----
    if name == "archaeology":
        query = args["query"]
        scope_handle = args.get("scope") or _resolve_scope_from_cwd_or_default()
        limit = int(args.get("limit", 5))
        # Three resolution paths in priority order: full id, 8-char prefix,
        # natural-language search. First two are cheap exact lookups; the
        # last falls through to recall() so the agent gets the full reading.
        traces: list[str] = []
        candidates: list = []
        # Try exact id
        full_id = _resolve_fragment_id(storage, query)
        if full_id is not None:
            frag = storage.get_fragment(full_id)
            if frag is not None:
                candidates = [frag]
        if not candidates:
            # Natural-language recall — agent will get the top-K traces
            from .models import RecallRequest
            from .retrieval import recall as do_recall
            req = RecallRequest(query=query, scope=scope_handle, limit=limit)
            resp = await asyncio.to_thread(do_recall, req, storage, provider)
            candidates = [r.fragment for r in resp.results[:limit]]
        if not candidates:
            return _tool_text(
                f"No fragments found for archaeology query {query!r}."
            )
        for frag in candidates[:limit]:
            chain: list[str] = [
                f"Fragment {frag.id[:8]}… [{frag.type}]",
                f"  Content: {frag.content[:160]}"
                + ("…" if len(frag.content) > 160 else ""),
                f"  Created: {frag.created_at} by {frag.created_by_tool or 'unknown'}",
            ]
            if frag.created_in_session_id:
                chain.append(f"  Session: {frag.created_in_session_id}")
            if frag.created_against_commit:
                chain.append(f"  Commit: {frag.created_against_commit}")
            if frag.supersedes_fragment_id:
                old = storage.get_fragment(frag.supersedes_fragment_id)
                if old:
                    chain.append(
                        f"  Supersedes {old.id[:8]}…: "
                        f"{old.content[:80]}"
                    )
            if frag.superseded_by_fragment_id:
                new = storage.get_fragment(frag.superseded_by_fragment_id)
                if new:
                    chain.append(
                        f"  Superseded by {new.id[:8]}…: "
                        f"{new.content[:80]}"
                    )
            chain.append(f"  Value: {frag.value:.2f}")
            traces.append("\n".join(chain))
        return _tool_text(
            f"Archaeology for {query!r} ({len(traces)} traces):\n\n"
            + "\n\n".join(traces)
        )

    # ---- query_leases ----
    if name == "query_leases":
        scope_handle = args.get("scope")
        scope_id = None
        if scope_handle:
            scope = storage.get_scope(scope_handle)
            if scope:
                scope_id = scope.id
        leases = storage.list_leases(scope_id=scope_id, active_only=True)
        if not leases:
            return _tool_text("No active leases.")
        lines = [f"Active leases ({len(leases)}):"]
        for lease in leases:
            lines.append(
                f"  {lease.id[:8]}… glob={lease.glob!r} "
                f"owner={lease.owner_id[:8]}… expires={lease.expires_at}"
            )
        return _tool_text("\n".join(lines))

    raise McpError(-32601, f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Resource definitions + reader
# ---------------------------------------------------------------------------

# URI templates (RFC 6570) exposed via `resources/templates/list`. The MCP
# spec separates concrete resources from templates so clients can prompt the
# user for the variable (here: ``scope``) before reading.
_RESOURCE_TEMPLATES = [
    {
        "uriTemplate": "context://{scope}/state",
        "name": "Current state",
        "description": "Latest state fragments for a scope",
        "mimeType": "text/plain",
    },
    {
        "uriTemplate": "context://{scope}/decisions",
        "name": "Active decisions",
        "description": "All active decision fragments for a scope",
        "mimeType": "text/plain",
    },
    {
        "uriTemplate": "context://{scope}/agents-md",
        "name": "AGENTS.md",
        "description": "Rendered AGENTS.md for a scope",
        "mimeType": "text/markdown",
    },
    {
        "uriTemplate": "context://{scope}/recent-commits",
        "name": "Recent commits",
        "description": "Last 20 commits for a scope",
        "mimeType": "text/plain",
    },
]


async def _read_resource(uri: str, storage: Any, request: Request) -> dict:
    from .agents_md import render_agents_md
    from .config import get_config

    cfg = get_config()

    # Parse context://{scope}/{type}
    if not uri.startswith("context://"):
        raise McpError(-32602, f"Unknown resource URI: {uri}")

    rest = uri[len("context://"):]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise McpError(-32602, "Invalid resource URI format")

    scope_handle, resource_type = parts[0], parts[1]

    scope = storage.get_scope(scope_handle)
    if not scope:
        return {"contents": [{"uri": uri, "mimeType": "text/plain",
                               "text": f"Scope '{scope_handle}' not found."}]}

    if resource_type == "state":
        frags = storage.list_fragments(scope_id=scope.id, type_filter="state",
                                        include_stale=False, limit=20)
        text = "\n\n".join(
            f"[{f.territory or '—'}] {f.content}" for f in frags
        ) or "No state fragments."
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    if resource_type == "decisions":
        frags = storage.list_fragments(scope_id=scope.id, type_filter="decision",
                                        include_stale=False, limit=50)
        text = "\n\n".join(
            f"• {f.content}" for f in frags
        ) or "No decision fragments."
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    if resource_type == "agents-md":
        text = render_agents_md(scope_handle, storage, daemon_url=cfg.base_url)
        return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": text}]}

    if resource_type == "recent-commits":
        commits = storage.list_commits(scope_id=scope.id, limit=20)
        lines = []
        for c in commits:
            lines.append(f"{c.created_at[:16]}  {c.id[:8]}  {c.message}")
        text = "\n".join(lines) or "No commits yet."
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    raise McpError(-32602, f"Unknown resource type: {resource_type}")


# ---------------------------------------------------------------------------
# Prompt definitions + getter
# ---------------------------------------------------------------------------

_PROMPTS = [
    {
        "name": "session_start",
        "description": (
            "Auto-inject AGENTS.md + top 5 relevant fragments at session start. "
            "Pass this prompt at the start of every agent session."
        ),
        "arguments": [
            {"name": "scope", "description": "Scope handle", "required": True},
            {"name": "task", "description": "What you're about to work on", "required": False},
        ],
    },
    {
        "name": "recall-first",
        "description": (
            "Mandatory retrieval rule. Inject this into the system prompt so the "
            "agent always queries Skein before answering project-specific "
            "questions instead of hallucinating."
        ),
        "arguments": [
            {"name": "scope", "description": "Scope handle", "required": False},
        ],
    },
]


_RECALL_FIRST_TEXT = """\
You have access to Skein, a shared context bus across every coding LLM \
working on this project. Other agents (Claude Code, Cursor, Codex, Gemini \
CLI, Antigravity, …) may have already stored decisions, observations, and \
codebase chunks here.

Rules — apply on every turn:

1. Before answering ANY question about this project's code, decisions, \
history, or architecture, call the `recall` tool first. Pass the user's \
question (or your task) as the query.
2. For code-level questions ("where is X defined?", "how does Y work?"), \
also call `search_code` to retrieve relevant codebase chunks.
3. If `recall` and `search_code` return nothing, say "I don't have context \
on that yet" — do not invent details.
4. After you make a non-trivial decision, finalize a plan, finish a task, \
or learn something the next agent will need, call `remember` (or \
`note_decision` for architectural choices) so future sessions inherit it.
5. Treat the returned fragments as authoritative project state. Prefer \
them over your prior assumptions.

Skein is the single source of truth for cross-session, cross-agent project \
context. Use it eagerly.
"""


def _build_recall_first_prompt(scope: str) -> dict:
    text = _RECALL_FIRST_TEXT
    if scope:
        text = (
            f"Active scope: {scope}\n\n" + text +
            f"\nPass `scope=\"{scope}\"` to recall/remember unless the user "
            "explicitly redirects you elsewhere."
        )
    return {
        "description": (
            f"Mandatory recall instruction"
            + (f" for scope {scope!r}" if scope else "")
        ),
        "messages": [
            {
                "role": "user",
                "content": {"type": "text", "text": text},
            }
        ],
    }


def _get_prompt(name: str, args: dict, storage: Any) -> dict:
    from .agents_md import render_agents_md
    from .config import get_config
    from .dependencies import get_provider as get_global_provider
    from .models import RecallRequest
    from .retrieval import recall as do_recall

    if name == "recall-first":
        return _build_recall_first_prompt(args.get("scope", ""))

    if name != "session_start":
        raise McpError(-32602, f"Unknown prompt: {name}")

    scope = args.get("scope", "")
    task = args.get("task", "")
    cfg = get_config()

    agents_md = render_agents_md(scope, storage, daemon_url=cfg.base_url)

    # Top 5 fragments relevant to the task
    context_note = ""
    if task and scope:
        try:
            # Re-use the daemon-wide provider singleton (initialised in
            # server.lifespan); building a fresh Gemini/OpenAI client per
            # prompt request adds ~hundreds of ms of cold-start.
            provider = get_global_provider()
            resp = do_recall(
                RecallRequest(query=task, scope=scope, limit=5),
                storage, provider,
            )
            if resp.results:
                lines = ["Relevant context for your task:\n"]
                for r in resp.results:
                    f = r.fragment
                    lines.append(f"- [{f.type}] {f.content}")
                context_note = "\n".join(lines)
        except Exception:
            pass

    prompt_text = agents_md
    if context_note:
        prompt_text += f"\n\n---\n\n{context_note}"

    return {
        "description": f"Session start context for scope {scope!r}",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": prompt_text,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _error_response(req_id: Any, code: int, message: str,
                    data: Optional[Any] = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


class McpError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
