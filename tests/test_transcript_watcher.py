"""Tests for the Claude Code transcript watcher (iter 14.2)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from skein.transcript_watcher import (
    ClaudeCodeTranscriptWatcher,
    decode_claude_project_dir,
    extract_from_text,
    parse_jsonl_line,
    transcripts_for_project,
)


# Iter 27 Windows port: the Claude Code project-directory encoding
# (`/Users/me/proj` → `-Users-me-proj`) was designed for POSIX paths and
# blows up on Windows where absolute paths contain a drive-letter colon
# (`C:\Users\me\proj` would encode to `-C:-Users-me-proj` — a colon is an
# illegal filename character on Windows, so the directory cannot be
# created). The transcript watcher is opt-in (SKEIN_TRANSCRIPT_WATCHER=1
# per HANDOFF.md); a Windows-aware encoding is documented as a follow-up.
# Path-round-trip + directory-creation tests below need a real watcher
# directory on disk, so they're skipped on Windows. The parsing /
# extraction tests above don't touch the filesystem and run everywhere.
_skip_on_windows_fs = pytest.mark.skipif(
    sys.platform.startswith("win") or os.name == "nt",
    reason=(
        "Transcript-watcher dir encoding (`-`-joined absolute path) is "
        "POSIX-only; Windows needs a drive-letter-safe encoding. "
        "Follow-up; watcher is opt-in."
    ),
)


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def test_parse_jsonl_user_message() -> None:
    line = json.dumps({
        "type": "user", "sessionId": "abc",
        "message": {"content": "hello world"},
        "timestamp": "2026-05-12T12:00:00",
        "uuid": "u1",
    })
    msg = parse_jsonl_line(line)
    assert msg is not None
    assert msg.role == "user"
    assert msg.text == "hello world"
    assert msg.session_id == "abc"


def test_parse_jsonl_assistant_with_blocks() -> None:
    line = json.dumps({
        "type": "assistant", "sessionId": "abc",
        "message": {"content": [
            {"type": "text", "text": "part 1"},
            {"type": "tool_use", "name": "Bash"},
            {"type": "text", "text": "part 2"},
        ]},
    })
    msg = parse_jsonl_line(line)
    assert msg is not None
    assert msg.role == "assistant"
    assert "part 1" in msg.text and "part 2" in msg.text


def test_parse_jsonl_skips_system_events() -> None:
    for t in ("system", "permission-mode", "file-history-snapshot", "ai-title"):
        assert parse_jsonl_line(json.dumps({"type": t})) is None


def test_parse_jsonl_skips_malformed() -> None:
    assert parse_jsonl_line("not json {{{") is None
    assert parse_jsonl_line("") is None


# ---------------------------------------------------------------------------
# Path encoding
# ---------------------------------------------------------------------------


@_skip_on_windows_fs
def test_decode_claude_project_dir_round_trip(tmp_path: Path) -> None:
    # Claude Code's encoding (replace ``/`` with ``-``) is irreversible if
    # the path itself contains hyphens. Pytest's tmp_path can contain hyphens
    # (e.g. ``pytest-31``), so we build a hyphen-free path under ``/tmp``
    # for the round-trip assertion.
    import os
    safe_root = Path("/tmp") / f"skein_test_{os.getpid()}"
    safe_dir = safe_root / "proj"
    safe_dir.mkdir(parents=True, exist_ok=True)
    try:
        encoded = "-" + str(safe_dir).lstrip("/").replace("/", "-")
        decoded = decode_claude_project_dir(encoded)
        assert decoded == safe_dir
    finally:
        import shutil
        shutil.rmtree(safe_root, ignore_errors=True)


def test_decode_returns_none_for_nonexistent_path() -> None:
    assert decode_claude_project_dir("-this-path-does-not-exist") is None


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------


def test_extract_lets_use_decision() -> None:
    facts = extract_from_text("let's use Redis for the session store", role="user")
    assert any("Redis" in f.content for f in facts)
    assert all(f.type == "decision" for f in facts if "Redis" in f.content)


def test_extract_remember_pattern() -> None:
    facts = extract_from_text(
        "Remember that the database is sharded by user_id and not by project_id.",
        role="user",
    )
    assert any("sharded" in f.content for f in facts)


def test_extract_todo_pattern() -> None:
    facts = extract_from_text(
        "TODO: migrate the auth table to use uuids before Friday",
        role="assistant",
    )
    assert any(f.type == "requirement" for f in facts)


def test_extract_preference_only_from_user() -> None:
    # The "i prefer" pattern is restricted to user role
    user_facts = extract_from_text("I prefer pytest fixtures with autouse=False", role="user")
    asst_facts = extract_from_text("I prefer pytest fixtures with autouse=False", role="assistant")
    assert any(f.type == "preference" for f in user_facts)
    assert all(f.type != "preference" for f in asst_facts)


def test_extract_strips_secrets() -> None:
    # We refuse to extract anything from messages containing secrets to
    # prevent partial-leak in fragment content.
    facts = extract_from_text(
        "Remember that our key is sk-abc1234567890123456789012345 — please save it",
        role="user",
    )
    assert facts == []


# ---------------------------------------------------------------------------
# Watcher cursor mechanics
# ---------------------------------------------------------------------------


@_skip_on_windows_fs
def test_transcripts_for_project_finds_jsonl(tmp_path: Path) -> None:
    # Set up a fake Claude Code root
    root = tmp_path / ".claude" / "projects"
    project = tmp_path / "myproj"
    project.mkdir()
    encoded = "-" + str(project).lstrip("/").replace("/", "-")
    proj_dir = root / encoded
    proj_dir.mkdir(parents=True)
    (proj_dir / "session-1.jsonl").write_text("{}")
    (proj_dir / "session-2.jsonl").write_text("{}")
    found = transcripts_for_project(project, root=root)
    assert len(found) == 2


@_skip_on_windows_fs
def test_watcher_poll_once_extracts_and_advances_cursor(tmp_path: Path) -> None:
    """End-to-end: write a transcript, poll, verify candidates landed and
    the cursor moved to EOF."""
    from skein.storage import Storage
    from skein.models import IdentityCreate, ScopeCreate
    from skein.embeddings import HashEmbeddingProvider

    db_path = tmp_path / "test.db"
    storage = Storage(str(db_path))
    ident = storage.get_or_create_identity(
        IdentityCreate(handle="user:t", type="user", name="t")
    )
    scope = storage.create_scope(
        ScopeCreate(handle="project:test", type="project",
                    name="test", owner_id=ident.id)
    )
    provider = HashEmbeddingProvider()

    # Build a fake project + transcript directory
    project = tmp_path / "proj"
    project.mkdir()
    root = tmp_path / ".claude" / "projects"
    encoded = "-" + str(project).lstrip("/").replace("/", "-")
    proj_dir = root / encoded
    proj_dir.mkdir(parents=True)
    transcript = proj_dir / "session-1.jsonl"
    msgs = [
        {"type": "user", "sessionId": "s1",
         "message": {"content": "let's use Postgres for the user table"}},
        {"type": "assistant", "sessionId": "s1",
         "message": {"content": [
             {"type": "text", "text": "Sure. TODO: drop the legacy mysql shard once migrated."}
         ]}},
        {"type": "system", "sessionId": "s1"},  # should be skipped
    ]
    transcript.write_text("\n".join(json.dumps(m) for m in msgs) + "\n")

    w = ClaudeCodeTranscriptWatcher(
        storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id,
        project_cwd=project, client_root=root,
        # Iter 32: default flipped to smart-only (≥0.90 conf). The seeded
        # messages here exercise the looser patterns (≥0.78), so explicitly
        # disable the filter for this watcher-mechanics test.
        smart_only=False,
    )
    n = w.poll_once()
    assert n >= 2  # parsed two non-system messages

    # Cursor should now equal file size
    cursor = storage.get_transcript_cursor(str(transcript))
    assert cursor == transcript.stat().st_size

    # Candidates or fragments produced (depending on confidence)
    candidates = storage.list_extraction_candidates(scope_id=scope.id, limit=20)
    frags = storage.list_fragments(scope_id=scope.id, limit=50)
    produced = len(candidates) + len(frags)
    assert produced >= 1

    # Second poll with no new content → 0 messages
    assert w.poll_once() == 0
    storage.close()
