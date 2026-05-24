"""Passive transcript watcher — extracts context from Claude Code chat logs.

iter 14.2: Claude Code persists every conversation as JSONL under
``~/.claude/projects/<encoded-project-path>/<session-uuid>.jsonl``. We tail
those files, run a fast heuristic extractor over each new user/assistant
message, and produce ``ScannedFact``-shaped candidates that the existing
``passive.promote_scanned_facts`` pipeline routes into fragments or the
review queue.

Why heuristic (regex) extraction rather than an LLM call?
- **Zero API cost.** Works offline; opt-in to upgrade to LLM extraction later.
- **Predictable.** ~5ms per message vs ~200ms+ network round-trip.
- **Privacy.** Nothing leaves the machine.

Trade-off: medium recall (catches the obvious "let's use X" patterns) but
high precision when it does catch. The review queue (``skein inbox``)
absorbs the false positives.

Cursor / Codex / Gemini-CLI transcript locations vary per platform and
aren't all stable yet — only Claude Code's path is reliable enough for v1.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .scanner import ScannedFact

logger = logging.getLogger("skein.transcript_watcher")


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def default_claude_code_root() -> Path:
    """Where Claude Code stores its JSONL transcripts."""
    return Path.home() / ".claude" / "projects"


def transcripts_for_project(cwd: Path,
                             root: Optional[Path] = None) -> list[Path]:
    """All transcript files for the given project working directory.

    Claude Code encodes the project path by replacing ``/`` with ``-`` and
    stripping the leading slash, so ``/Users/ameliomar/Documents/foo`` maps
    to ``-Users-ameliomar-Documents-foo``. We look up exactly that folder
    and return every JSONL inside.
    """
    root = root or default_claude_code_root()
    encoded = "-" + str(cwd.resolve()).lstrip("/").replace("/", "-")
    project_dir = root / encoded
    if not project_dir.is_dir():
        return []
    return sorted(project_dir.glob("*.jsonl"))


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


@dataclass
class ParsedMessage:
    role: str           # "user" | "assistant"
    text: str
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    uuid: Optional[str] = None


def parse_jsonl_line(raw: str) -> Optional[ParsedMessage]:
    """Pull the text content out of one Claude Code JSONL line.

    Returns None for non-text events (file snapshots, permission mode,
    tool calls, system events, etc.). Survives malformed lines.
    """
    try:
        d = json.loads(raw)
    except Exception:
        return None
    t = d.get("type")
    if t not in ("user", "assistant"):
        return None
    msg = d.get("message") or {}
    content = msg.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Multi-block messages: take all text blocks concatenated
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    text = text.strip()
    if not text:
        return None
    return ParsedMessage(
        role=t,
        text=text,
        timestamp=d.get("timestamp"),
        session_id=d.get("sessionId"),
        uuid=d.get("uuid"),
    )


# ---------------------------------------------------------------------------
# Heuristic extractor — regex patterns → candidate facts
# ---------------------------------------------------------------------------


@dataclass
class ExtractionPattern:
    """One regex-based extractor."""
    pattern: re.Pattern
    type: str                # decision | fact | preference | requirement | procedure
    confidence: float
    # Group index whose match becomes the fragment content (1-indexed).
    content_group: int = 1
    # Optional prefix added in front of the matched content.
    content_prefix: str = ""
    # Roles this pattern applies to.
    roles: tuple = ("user", "assistant")


# ---------------------------------------------------------------------------
# Built-in pattern set
#
# We bias toward patterns that:
#  - have an explicit subject phrase ("let's use", "we decided")
#  - state a concrete noun-phrase or sentence afterward
#  - aren't too generic (e.g. plain "use X" would match too much)
#
# Confidences are deliberate: 0.85+ auto-promotes; 0.6-0.84 → inbox; <0.5
# discards. Tune via testing.
# ---------------------------------------------------------------------------

_PATTERNS: list[ExtractionPattern] = [
    # --- iter 32: high-precision auto-promote patterns ---
    # These catch the load-bearing sentences a coding LLM writes in an iter
    # recap but routinely forgets to `note_decision` back to Skein. Confidence
    # ≥ AUTO_PROMOTE_THRESHOLD (0.90) so they land as fragments without
    # waiting in the inbox. Each pattern starts with an unambiguous signal
    # phrase (capital-I "Iter N SHIPPED", "Decided to X", "Concluded that X")
    # — false-positive rate in regular prose is near zero.
    ExtractionPattern(
        pattern=re.compile(
            r"\bIter\s+\d+(?:\.\d+)?\s+(?:SHIPPED|shipped|complete|completed|in progress|opened|landed|done)\b[\s\-—:]*[^\n]{10,400}",
        ),
        type="decision", confidence=0.92,
        content_group=0,
    ),
    ExtractionPattern(
        pattern=re.compile(
            r"\bDecided to\s+(?:use|go with|pick|choose|switch to|migrate to|drop|adopt|deprecate|ship|skip|defer)\s+([^\.\n]{4,200})",
            re.IGNORECASE,
        ),
        type="decision", confidence=0.92,
        content_prefix="Decided to ",
    ),
    ExtractionPattern(
        pattern=re.compile(
            r"\bConcluded(?:\s+that)?\s+([^\.\n]{8,300})",
            re.IGNORECASE,
        ),
        type="decision", confidence=0.90,
        content_prefix="Concluded: ",
    ),
    ExtractionPattern(
        pattern=re.compile(
            r"\b(?:Shipped|SHIPPED)\s+(iter\s+\d+(?:\.\d+)?|the\s+\w[\w\-]{2,40}(?:\s+\w[\w\-]{2,40}){0,3})[\s\-—:]*([^\n]{10,300})?",
        ),
        type="decision", confidence=0.90,
        content_group=0,
    ),
    # --- legacy lower-confidence patterns (inbox-bound; opt-in via env) ---
    # "let's use X" / "let us use X" — high signal, user-stated decision
    ExtractionPattern(
        pattern=re.compile(
            r"\b(?:let'?s|let us)\s+(?:use|go with|pick|choose|switch to|migrate to)\s+([^\.\n,;]{4,120})",
            re.IGNORECASE,
        ),
        type="decision", confidence=0.85,
        content_prefix="Decided to use ",
    ),
    # "we decided X" / "we'll go with X"
    ExtractionPattern(
        pattern=re.compile(
            r"\bwe(?:'?ll|'?re going to| will| decided to|'?ve decided to)\s+([a-z][^\.\n]{8,160})",
            re.IGNORECASE,
        ),
        type="decision", confidence=0.78,
        content_prefix="Decision: ",
    ),
    # "i prefer X" / "always use X" / "never use X"
    ExtractionPattern(
        pattern=re.compile(
            r"\b(?:i prefer|i like|i want|always use|never use|always|prefer to)\s+([^\.\n]{6,140})",
            re.IGNORECASE,
        ),
        type="preference", confidence=0.72,
        content_prefix="Preference: ",
        roles=("user",),  # only user-stated preferences
    ),
    # "remember (that) X" / "note (that) X"
    ExtractionPattern(
        pattern=re.compile(
            r"\b(?:remember(?:\s+that)?|note(?:\s+that)?|keep in mind(?:\s+that)?)\b\s*[:,]?\s+([A-Z][^\.\n]{8,200})",
            re.IGNORECASE,
        ),
        type="fact", confidence=0.86,
        content_prefix="",
    ),
    # "the X is Y" — slightly weaker, queue rather than auto
    ExtractionPattern(
        pattern=re.compile(
            r"\bthe\s+([a-z][a-zA-Z0-9_\- ]{3,40})\s+is\s+([a-zA-Z0-9_\-][^\.\n]{4,140})",
            re.IGNORECASE,
        ),
        type="fact", confidence=0.62,
        content_group=0,  # use full match
    ),
    # "TODO: X" / "FIXME: X" — code-comment-style notes
    ExtractionPattern(
        pattern=re.compile(
            r"\b(?:TODO|FIXME)\b\s*[:\-]?\s+([^\n]{6,200})",
        ),
        type="requirement", confidence=0.78,
        content_prefix="TODO: ",
    ),
    # "to deploy, run X" / procedure
    ExtractionPattern(
        pattern=re.compile(
            r"\bto\s+(deploy|build|release|test|run|start)[^,.\n]{0,40},?\s*(?:run|use|execute|call)\s+([^\.\n]{4,160})",
            re.IGNORECASE,
        ),
        type="procedure", confidence=0.74,
        content_group=0,
    ),
]


# Boilerplate that often shows up inside captured groups but isn't worth storing.
_BOILERPLATE_FRAGMENTS = (
    "the same",
    "that one",
    "this one",
    "it",
    "the thing",
    "the file",
    "the code",
)


# Lightweight secret-pattern scrubbing. Better than nothing; not a substitute
# for real secret-detection in v2.
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),                # OpenAI-style
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                   # AWS access key
    re.compile(r"\bgh[ps]_[A-Za-z0-9]{30,}\b"),            # GitHub token
    re.compile(r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),  # JWT-ish
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{20,}\b"),
]


def _scrub_secrets(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED-SECRET]", text)
    return text


def extract_from_message(
    msg: ParsedMessage, *, smart_only: bool = False,
) -> list[ScannedFact]:
    """Run every pattern over one parsed message. Returns 0+ candidate facts.

    ``smart_only=True`` restricts to the high-precision (confidence ≥ 0.90)
    pattern set. Iter 32 default for the live transcript watcher so the loose
    patterns (which generate inbox noise) only fire when the user opts in via
    ``SKEIN_TRANSCRIPT_WATCHER=loose``.
    """
    out: list[ScannedFact] = []
    text = _scrub_secrets(msg.text)
    if "[REDACTED-SECRET]" in text:
        # We don't try to extract from messages that contained secrets; risk
        # of leaking a partial credential into a fragment isn't worth the
        # marginal recall.
        return out
    seen_contents: set = set()
    for spec in _PATTERNS:
        if smart_only and spec.confidence < 0.90:
            continue
        if msg.role not in spec.roles:
            continue
        for m in spec.pattern.finditer(text):
            try:
                core = m.group(spec.content_group)
            except IndexError:
                continue
            core = core.strip(" \t.,;:")
            if not core or len(core) < 4:
                continue
            lowered = core.lower()
            if lowered in _BOILERPLATE_FRAGMENTS:
                continue
            content = (spec.content_prefix + core).strip()
            if content.lower() in seen_contents:
                continue
            seen_contents.add(content.lower())
            out.append(ScannedFact(
                content=content,
                type=spec.type,
                confidence=spec.confidence,
                source_file=None,
                tags=["passive", "transcript"],
            ))
    return out


def extract_from_text(
    text: str, role: str = "user", *, smart_only: bool = False,
) -> list[ScannedFact]:
    """Convenience wrapper used by tests."""
    return extract_from_message(
        ParsedMessage(role=role, text=text), smart_only=smart_only,
    )


# ---------------------------------------------------------------------------
# Watcher — reads transcripts from disk, hands off to passive.promote_*
# ---------------------------------------------------------------------------


class ClaudeCodeTranscriptWatcher:
    """Polling watcher for Claude Code JSONL transcripts.

    Maintains per-file byte-offset cursors in the ``transcript_cursors``
    table so it resumes cleanly across daemon restarts and ignores already-
    processed messages.

    Single-threaded by design: a background daemon thread polls every
    ``poll_interval`` seconds; each poll reads new bytes from every open
    transcript and promotes candidates via the supplied callback.
    """

    def __init__(
        self,
        *,
        storage,
        provider,
        scope_id: str,
        owner_id: str,
        project_cwd: Path,
        poll_interval: float = 2.0,
        client_root: Optional[Path] = None,
        source_tool: str = "transcript-claude",
        smart_only: bool = True,
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.scope_id = scope_id
        self.owner_id = owner_id
        self.project_cwd = Path(project_cwd).resolve()
        self.poll_interval = poll_interval
        self.client_root = client_root or default_claude_code_root()
        self.source_tool = source_tool
        self.smart_only = smart_only
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="skein-transcript-watcher", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ---- core loop ----

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.debug("transcript watcher poll failed", exc_info=True)
            self._stop.wait(self.poll_interval)

    def poll_once(self) -> int:
        """Process any unseen bytes across all transcripts. Returns the
        number of new messages extracted (for tests / diagnostics).
        """
        files = transcripts_for_project(self.project_cwd, root=self.client_root)
        new_msgs = 0
        for path in files:
            new_msgs += self._poll_file(path)
        return new_msgs

    def _poll_file(self, path: Path) -> int:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return 0
        cursor = self.storage.get_transcript_cursor(str(path))
        if cursor >= size:
            return 0
        # If the file shrank (rotated/truncated), reset to 0
        if cursor > size:
            cursor = 0
        new_msgs = 0
        new_cursor = cursor
        candidates: list[ScannedFact] = []
        try:
            with open(path, "rb") as f:
                f.seek(cursor)
                buf = f.read()
                new_cursor = cursor + len(buf)
        except OSError:
            return 0
        # Parse line-by-line; tolerate partial final line by trimming to last \n
        last_newline = buf.rfind(b"\n")
        if last_newline < 0:
            # No complete line yet; wait for next poll
            return 0
        complete = buf[: last_newline + 1]
        new_cursor = cursor + last_newline + 1
        for raw_line in complete.split(b"\n"):
            if not raw_line.strip():
                continue
            try:
                msg = parse_jsonl_line(raw_line.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if not msg:
                continue
            new_msgs += 1
            for fact in extract_from_message(msg, smart_only=self.smart_only):
                # Tag candidate with the originating session so archaeology
                # can trace it back later.
                fact.tags = list(set(fact.tags + (["session:" + msg.session_id] if msg.session_id else [])))
                candidates.append(fact)

        if candidates:
            # Use the passive promotion pipeline; it handles dedup + commits
            from .passive import promote_scanned_facts
            try:
                promote_scanned_facts(
                    candidates, storage=self.storage, provider=self.provider,
                    scope_id=self.scope_id, owner_id=self.owner_id,
                    source_tool=self.source_tool,
                )
            except Exception:
                logger.debug("promote_scanned_facts failed", exc_info=True)

        # Save cursor even when nothing was extracted, so we don't re-scan.
        self.storage.set_transcript_cursor(
            str(path), new_cursor, client_name="claude-code",
        )
        return new_msgs


# ---------------------------------------------------------------------------
# Multi-project orchestrator — what the daemon actually runs
# ---------------------------------------------------------------------------


def decode_claude_project_dir(name: str) -> Optional[Path]:
    """Reverse the Claude Code path encoding.

    ``-Users-ameliomar-Documents-foo`` → ``/Users/ameliomar/Documents/foo``.
    Returns None if the directory doesn't exist on disk (stale/old project).
    """
    if not name.startswith("-"):
        return None
    candidate = Path("/" + name[1:].replace("-", "/"))
    if candidate.exists():
        return candidate
    return None


class MultiProjectTranscriptWatcher:
    """Polls every Claude Code project directory and processes new transcripts.

    For each ``~/.claude/projects/<dir>/*.jsonl``, decodes the project path,
    looks up the matching Skein scope (by ``.skein/scope`` pin OR by
    ``project:<basename>`` convention OR by scope_resolver), and if a scope
    exists, runs the extractor against any unseen bytes.

    Projects with no Skein scope are silently skipped — opt-in by user.
    """

    def __init__(
        self,
        *,
        storage_factory,        # callable that returns a fresh Storage handle
        provider,
        poll_interval: float = 3.0,
        client_root: Optional[Path] = None,
        get_owner_id,           # callable() -> identity id
        smart_only: bool = True,
    ) -> None:
        self.storage_factory = storage_factory
        self.provider = provider
        self.poll_interval = poll_interval
        self.client_root = client_root or default_claude_code_root()
        self.get_owner_id = get_owner_id
        self.smart_only = smart_only
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="skein-multi-transcript-watcher", daemon=True,
        )
        self._thread.start()
        logger.info("transcript watcher started; root=%s", self.client_root)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.debug("multi watcher poll failed", exc_info=True)
            self._stop.wait(self.poll_interval)

    def poll_once(self) -> dict[str, int]:
        """Returns ``{project_path: new_messages_processed}`` for telemetry."""
        out: dict[str, int] = {}
        if not self.client_root.is_dir():
            return out
        storage = self.storage_factory()
        try:
            owner_id = self.get_owner_id(storage)
            for entry in sorted(self.client_root.iterdir()):
                if not entry.is_dir():
                    continue
                project_path = decode_claude_project_dir(entry.name)
                if not project_path:
                    continue
                scope_id = self._resolve_scope_id(storage, project_path)
                if not scope_id:
                    continue
                w = ClaudeCodeTranscriptWatcher(
                    storage=storage, provider=self.provider,
                    scope_id=scope_id, owner_id=owner_id,
                    project_cwd=project_path,
                    client_root=self.client_root,
                    smart_only=self.smart_only,
                )
                n = w.poll_once()
                if n:
                    out[str(project_path)] = n
        finally:
            try:
                storage.close()
            except Exception:
                pass
        return out

    def _resolve_scope_id(self, storage, project_path: Path) -> Optional[str]:
        """Look up the scope this project maps to.

        Order:
          1. ``.skein/scope`` pin file in the project root
          2. ``project:<basename>`` if it already exists in the scopes table
        """
        pin = project_path / ".skein" / "scope"
        scope_handle: Optional[str] = None
        if pin.is_file():
            try:
                scope_handle = pin.read_text().strip()
            except OSError:
                pass
        if not scope_handle:
            # Don't auto-create scopes here — only watch projects the user
            # explicitly initialized with `skein up`.
            return None
        scope = storage.get_scope(scope_handle)
        return scope.id if scope else None
