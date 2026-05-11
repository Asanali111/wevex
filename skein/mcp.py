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
from typing import Any, Dict, List, Optional

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

async def _handle_one(msg: Dict[str, Any], request: Request) -> Optional[Dict]:
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


async def _dispatch(method: str, params: Dict[str, Any], request: Request) -> Any:
    from .dependencies import get_provider, get_storage

    storage = get_storage()
    provider = get_provider()

    # ---- Lifecycle ----
    if method == "initialize":
        return _handle_initialize(params)

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
        name = params.get("name")
        args = params.get("arguments") or {}
        return _get_prompt(name, args, storage)

    raise McpError(-32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

def _handle_initialize(params: Dict) -> Dict:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": {"name": "skein", "version": "0.1.0"},
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"listChanged": False, "subscribe": False},
            "prompts": {"listChanged": False},
        },
    }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "recall",
        "description": (
            "Search for relevant context fragments from the Skein context bus. "
            "Use this at the start of any task to load relevant decisions, "
            "state, requirements, and preferences. Returns ranked fragments. "
            "The `scope` argument is optional — Skein auto-detects it from "
            "the daemon's working directory (project pin / git remote)."
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
        "description": "Retrieve the full content of a specific fragment by ID (progressive disclosure).",
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
            "Store a context fragment in the Skein context bus. "
            "Call this after any significant decision, observation, or state change. "
            "Other agents and tools will be able to recall this context. "
            "The `scope` argument is optional — Skein auto-detects from cwd."
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
            "Record an architectural or technical decision. "
            "Convenience wrapper around remember(type='decision') with structured alternatives/rationale."
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
            "Acquire an advisory lease on a file-glob pattern. "
            "Informs other agents that you are working on this area. "
            "They will see a LEASE_CONFLICT if they try to acquire the same glob."
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
        "description": "Release a previously acquired advisory lease.",
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
        "description": "List active advisory leases for a scope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search the project's indexed codebase / documents for relevant chunks. "
            "Returns code snippets with file paths and line ranges. "
            "Use this when you need to find existing functions, types, or "
            "documentation by semantic meaning, not just text match. "
            "Code must be ingested first via `skein ingest` before this works. "
            "The `scope` argument is optional — Skein auto-detects from cwd."
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
]


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
# Tool call handler
# ---------------------------------------------------------------------------

async def _call_tool(
    name: str,
    args: Dict[str, Any],
    storage: Any,
    provider: Any,
    request: Request,
) -> Dict:
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

    # ---- recall ----
    if name == "recall":
        req = RecallRequest(
            query=args["query"],
            scope=args["scope"],
            types=args.get("types"),
            territory=args.get("territory"),
            limit=args.get("limit", 10),
        )
        response = do_recall(req, storage, provider)
        if not response.results:
            return _tool_text("No relevant context found.")
        lines = [f"Found {response.total} fragments for query: {response.query!r}\n"]
        for r in response.results:
            f = r.fragment
            territory_note = f" [{f.territory}]" if f.territory else ""
            tags_note = f" #{' #'.join(f.tags)}" if f.tags else ""
            lines.append(
                f"[{r.rank}] {f.type.upper()}{territory_note}{tags_note} "
                f"(score={r.score:.3f}, id={f.id[:8]}…)\n"
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
        )

        from .embeddings import vec_to_bytes
        from .models import CommitCreate
        embedding_bytes = None
        try:
            vec = provider.embed_one(args["content"])
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
            vec = provider.embed_one(full_content)
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
        )
        frag = storage.create_fragment(data, commit_id=commit.id, embedding=embedding_bytes)
        storage._conn.execute(
            "UPDATE commits SET fragments_added = ? WHERE id = ?",
            (f'["{frag.id}"]', commit.id),
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
        return _tool_text(
            f"Lease acquired: {lease.id[:8]}… on '{lease.glob}' "
            f"(expires {lease.expires_at})"
        )

    # ---- release_lease ----
    if name == "release_lease":
        released = storage.release_lease(args["lease_id"], owner_id)
        if released:
            return _tool_text(f"Lease {args['lease_id'][:8]}… released.")
        return _tool_text("Lease not found or not owned by you.")

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
        response = search_chunks(req, storage, provider)
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
            lines.append(
                f"[{r.rank}] {c.source_path}:{c.line_start}-{c.line_end}{lang}{sym}  "
                f"score={r.score:.3f}\n"
                f"```{c.language or ''}\n{c.content}\n```\n"
            )
        return _tool_text("\n".join(lines))

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


async def _read_resource(uri: str, storage: Any, request: Request) -> Dict:
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


def _build_recall_first_prompt(scope: str) -> Dict:
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


def _get_prompt(name: str, args: Dict, storage: Any) -> Dict:
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

def _tool_text(text: str) -> Dict:
    return {"content": [{"type": "text", "text": text}]}


def _error_response(req_id: Any, code: int, message: str,
                    data: Optional[Any] = None) -> Dict:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


class McpError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
