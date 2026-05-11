"""Hook handlers — these are what Claude Code (and similar) actually invoke.

Each handler is a thin function that:
  1. Reads JSON input from stdin (per the hook spec for that client)
  2. Talks to the local Skein storage directly (no HTTP — daemon may be off)
  3. Writes the appropriate output to stdout

Direct storage access lets hooks run with sub-100ms latency and still work
when the daemon isn't running (the SQLite DB is shared via WAL mode).

Supported hook events (Claude Code naming):
  SessionStart        Inject project context at the start of a session.
  UserPromptSubmit    Inject extra context based on the user's prompt.
  Stop                Capture decisions/observations from the assistant turn.
  PostToolUse         Capture file-change observations after Edit/Write tools.

The Cursor / Codex / Gemini equivalents are passive (rules / AGENTS.md);
they call the same `skein recall` / `skein remember` CLI rather than these
hook handlers, so this module is Claude Code-shaped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .embeddings import get_provider as get_embedding_provider
from .models import (
    FragmentCreate, IdentityCreate, RecallRequest, ScopeCreate,
)
from .retrieval import recall as do_recall
from .storage import Storage

logger = logging.getLogger("skein.hooks")

# How many fragments to inject at SessionStart
SESSION_START_LIMIT = 8
# How many to inject on UserPromptSubmit (smaller — runs every turn)
USER_PROMPT_LIMIT = 5
# Min RRF score required for a hit to be injected (filters out 0.005-tier noise).
# RRF scores top out around 0.033 for a perfect 1st-place match in two lists,
# so 0.025 means "appeared near the top of at least one list".
MIN_INJECT_SCORE = 0.025

# Types that carry signal vs. types that are mostly self-generated noise.
# SessionStart only injects signal types; UserPromptSubmit allows everything
# but score threshold filters out the noise.
SIGNAL_TYPES = {"decision", "requirement", "preference", "fact", "procedure"}
# Order in which sections appear in the injected markdown.
SECTION_ORDER = ["requirement", "decision", "preference", "fact",
                 "procedure", "state", "observation", "conversation"]

# Decision-detection regex patterns (used by the Stop hook)
_DECISION_PATTERNS = [
    r"\bdecided?\s+to\b",
    r"\blet'?s\s+(use|go\s+with|pick|choose)\b",
    r"\bwe'?ll\s+(use|go\s+with|pick|choose)\b",
    r"\bgoing\s+with\b",
    r"\bI'?ll\s+(use|implement|build|switch\s+to)\b",
    r"\bswitching?\s+to\b",
    r"\bchose\b",
]
_DECISION_RE = re.compile("|".join(_DECISION_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _open_storage_and_scope() -> tuple[Storage, str]:
    """Open storage and resolve the active scope handle for cwd.

    Delegates to the shared ``scope_resolver`` so hooks and CLI commands
    follow the same precedence rules.
    """
    from .scope_resolver import resolve_scope

    cfg = get_config()
    storage = Storage(cfg.db_path)
    scope, _source = resolve_scope(None, config_default=cfg.default_scope)

    # Auto-create scope and a default "user:local" identity if missing
    s = storage.get_scope(scope)
    if not s:
        # Need an owner; auto-create a service identity
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="agent:claude-code-hook",
            type="agent",
            name="Claude Code (hook)",
        ))
        storage.create_scope(ScopeCreate(
            handle=scope,
            type=_infer_scope_type(scope),
            name=scope.split(":", 1)[-1],
            owner_id=owner.id,
        ))

    return storage, scope


def _infer_scope_type(handle: str) -> str:
    """Derive scope type from the handle prefix, defaulting to project."""
    prefix = handle.split(":", 1)[0]
    return prefix if prefix in {"public", "org", "team", "project", "personal"} else "project"


def _author_identity(storage: Storage, agent_label: str = "claude-code") -> str:
    """Get-or-create an agent identity for the calling tool."""
    handle = f"agent:{agent_label}"
    ident = storage.get_or_create_identity(IdentityCreate(
        handle=handle,
        type="agent",
        name=agent_label,
    ))
    return ident.id


# ---------------------------------------------------------------------------
# SessionStart
# ---------------------------------------------------------------------------

def session_start(stdin_text: str = "") -> int:
    """Run the SessionStart hook.

    Output format (Claude Code SessionStart):
      Print plain text → injected as additional system context.
      Or JSON {"hookSpecificOutput": {"hookEventName": "SessionStart",
              "additionalContext": "<text>"}}

    Only emits if there's signal worth injecting — bare observation logs
    ("Edit on cli.py") never make it through. Empty result → empty output
    (no padding, no apology).
    """
    try:
        storage, scope_handle = _open_storage_and_scope()
    except Exception as e:
        logger.warning("session_start init failed: %s", e)
        return 0

    try:
        scope = storage.get_scope(scope_handle)
        if not scope:
            storage.close()
            return 0

        # Pull only signal-bearing types. Drop observations / conversations
        # entirely from session-start — they're auto-generated logs, not
        # context the next agent needs.
        signal_frags = []
        for ftype in SECTION_ORDER:
            if ftype not in SIGNAL_TYPES:
                continue
            rows = storage.list_fragments(
                scope_id=scope.id, type_filter=ftype,
                include_stale=False, limit=20,
            )
            signal_frags.extend(rows)

        # Order: section priority asc, then recency desc inside each section.
        signal_frags.sort(
            key=lambda f: (SECTION_ORDER.index(f.type), -_ts(f.updated_at)),
        )

        # Dedupe by content so identical fragments don't appear multiple times.
        seen_contents = set()
        deduped = []
        for f in signal_frags:
            key = f.content.strip().lower()[:200]
            if key in seen_contents:
                continue
            seen_contents.add(key)
            deduped.append(f)

        top = deduped[:SESSION_START_LIMIT]
        if not top:
            # No signal — say nothing rather than pad with noise.
            storage.close()
            return 0

        text = _render_grouped(scope_handle, top, header="Skein context")
        out = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": text,
            }
        }
        print(json.dumps(out))
        storage.close()
        return 0
    except Exception as e:
        logger.exception("session_start failed: %s", e)
        return 0


def _render_grouped(scope_handle: str, frags: List, *, header: str) -> str:
    """Group fragments by type and render compact markdown.

    The output the AI actually sees on every prompt — keep it clean."""
    by_type: Dict[str, List] = {}
    for f in frags:
        by_type.setdefault(f.type, []).append(f)

    lines: List[str] = [f"## {header} — `{scope_handle}`", ""]
    for ftype in SECTION_ORDER:
        bucket = by_type.get(ftype)
        if not bucket:
            continue
        lines.append(f"**{ftype.capitalize()}s**")
        for f in bucket:
            terr = f" _({f.territory})_" if f.territory else ""
            lines.append(f"- {f.content}{terr}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# UserPromptSubmit
# ---------------------------------------------------------------------------

def user_prompt_submit(stdin_text: str = "") -> int:
    """Hook input is JSON; we want the prompt text, then recall against it."""
    try:
        data = json.loads(stdin_text) if stdin_text.strip() else {}
    except json.JSONDecodeError:
        data = {}

    prompt = (
        data.get("prompt")
        or data.get("user_prompt")
        or data.get("message")
        or stdin_text  # fallback: treat entire stdin as the prompt
    ).strip()
    if not prompt or len(prompt) < 5:
        return 0

    try:
        storage, scope_handle = _open_storage_and_scope()

        # Fast path: scope has zero fragments → nothing to recall, return now.
        # Avoids the 50–200ms cost of importing the embedding provider, embedding
        # the query, and running BM25 + vector for an empty result set.
        scope = storage.get_scope(scope_handle)
        if scope is None or storage.count_fragments_in_scope(scope.id) == 0:
            storage.close()
            return 0

        provider = _get_provider_safe()
        if provider is None:
            storage.close()
            return 0

        req = RecallRequest(query=prompt[:500], scope=scope_handle, limit=USER_PROMPT_LIMIT)
        resp = do_recall(req, storage, provider)
        storage.close()
    except Exception as e:
        logger.exception("user_prompt_submit failed: %s", e)
        return 0

    # Filter low-score hits to avoid injecting irrelevant noise
    hits = [r for r in resp.results if r.score >= MIN_INJECT_SCORE]

    # Dedupe by content — RRF can otherwise return the same fragment twice
    # if it ranked in both the keyword and vector lists (it shouldn't, but
    # bare observation rows with near-identical content slip through).
    seen = set()
    deduped = []
    for r in hits:
        key = r.fragment.content.strip().lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    hits = deduped

    # Drop pure observation/conversation hits unless they have a real score.
    # 0.04+ in RRF means "near the top of one list" so we let them through;
    # otherwise observations are tool-event logs that just clutter the prompt.
    OBS_FLOOR = 0.04
    hits = [
        r for r in hits
        if r.fragment.type in SIGNAL_TYPES or r.score >= OBS_FLOOR
    ]
    if not hits:
        return 0

    frags = [r.fragment for r in hits]
    text = _render_grouped(
        scope_handle, frags,
        header=f"Skein recall — `{prompt[:60]}…`",
    )
    out = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        }
    }
    print(json.dumps(out))
    return 0


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

def stop(stdin_text: str = "") -> int:
    """The Stop hook fires when Claude finishes a turn.

    We scan the assistant's last response for decision-shaped sentences and
    persist them as `decision` fragments.
    """
    try:
        data = json.loads(stdin_text) if stdin_text.strip() else {}
    except json.JSONDecodeError:
        data = {}

    # Try various keys Claude Code might use
    transcript = (
        data.get("transcript")
        or data.get("response")
        or data.get("assistant_message")
        or ""
    )
    if isinstance(transcript, list):
        # If it's a list of messages, pick the last assistant one
        for m in reversed(transcript):
            if isinstance(m, dict) and m.get("role") == "assistant":
                transcript = m.get("content", "")
                break
        else:
            transcript = ""

    if not isinstance(transcript, str):
        transcript = str(transcript)

    decisions = _extract_decisions(transcript)
    if not decisions:
        return 0

    try:
        storage, scope_handle = _open_storage_and_scope()
        scope = storage.get_scope(scope_handle)
        if scope is None:
            storage.close()
            return 0

        author_id = _author_identity(storage, "claude-code")
        provider = _get_provider_safe()

        for sentence in decisions[:5]:  # cap at 5 per turn
            payload = FragmentCreate(
                type="decision",
                content=sentence,
                scope_id=scope.id,
                owner_id=author_id,
                tags=["auto-extracted"],
                metadata={"source": "claude-code-stop-hook"},
            )
            embedding_bytes: Optional[bytes] = None
            if provider:
                try:
                    from .embeddings import vec_to_bytes
                    embedding_bytes = vec_to_bytes(provider.embed_one(sentence))
                except Exception:
                    pass
            storage.create_fragment(payload, embedding=embedding_bytes)

        storage.close()
    except Exception as e:
        logger.exception("stop failed: %s", e)

    return 0


# ---------------------------------------------------------------------------
# PostToolUse
# ---------------------------------------------------------------------------

def post_tool_use(stdin_text: str = "") -> int:
    """Captured-but-quiet: file edits are NOT stored as fragments anymore.

    The previous behaviour ("Edit on /Users/.../cli.py") created identical,
    contentless observation rows that polluted both the DB and every future
    SessionStart injection. From the AI consumer's POV those rows carry zero
    signal — the file path is already visible to the agent that just edited
    the file. We keep the hook wired so client config doesn't break, but it's
    a no-op now.

    If we want this back later, do it with content (a one-line diff summary,
    or the function name extracted from the patch), not just the path.
    """
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_decisions(text: str) -> List[str]:
    """Return sentences that look like decisions."""
    if not text:
        return []
    # Split into rough sentences
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    out: List[str] = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if 15 <= len(s) <= 280 and _DECISION_RE.search(s):
            key = s.lower()
            if key not in seen:
                out.append(s)
                seen.add(key)
    return out


def _ts(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _path_to_territory(file_path: str) -> Optional[str]:
    """Map a file path to a territory (top two path components)."""
    parts = Path(file_path).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}".rstrip("/")
    elif parts:
        return parts[0]
    return None


def _get_provider_safe():
    """Return embedding provider or None on any failure (offline-first)."""
    try:
        cfg = get_config()
        return get_embedding_provider(cfg.embedding_provider)
    except Exception as e:
        logger.warning("embedding provider unavailable, recall will skip vector: %s", e)
        return None
