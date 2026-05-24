"""Tests for the JSONL event log (R-02)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skein.events import EventLogger, MAX_BYTES, default_path, log_event, reset_event_logger


@pytest.fixture
def tmp_events(tmp_path, monkeypatch):
    """Point the events singleton at a fresh file under tmp_path."""
    p = tmp_path / "events.jsonl"
    monkeypatch.setenv("SKEIN_EVENTS_PATH", str(p))
    reset_event_logger()
    yield p
    reset_event_logger()


def test_log_writes_jsonl_line(tmp_events: Path) -> None:
    log_event("recall", scope="project:test", query="auth", hits=3)
    assert tmp_events.exists()
    lines = tmp_events.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "recall"
    assert rec["scope"] == "project:test"
    assert rec["details"]["query"] == "auth"
    assert rec["details"]["hits"] == 3
    assert "ts" in rec


def test_log_never_raises_on_bad_path(tmp_path, monkeypatch) -> None:
    """log_event must swallow filesystem errors — never break recall/remember."""
    # Point at a path under a regular file (so mkdir will fail)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("SKEIN_EVENTS_PATH", str(blocker / "events.jsonl"))
    reset_event_logger()
    # Should not raise even though parent isn't a real dir
    log_event("recall", scope="project:x", query="anything")
    reset_event_logger()


def test_log_appends_multiple(tmp_events: Path) -> None:
    log_event("recall", scope="s1", query="q1", hits=1)
    log_event("remember", scope="s1", fragment_id="abc", type="fact")
    log_event("supersede", scope="s1", old_fragment_id="a", new_fragment_id="b")
    lines = tmp_events.read_text().strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(l)["event"] for l in lines]
    assert events == ["recall", "remember", "supersede"]


def test_log_rotation(tmp_path, monkeypatch) -> None:
    """When the file exceeds MAX_BYTES, it rotates to .1."""
    p = tmp_path / "events.jsonl"
    monkeypatch.setenv("SKEIN_EVENTS_PATH", str(p))
    reset_event_logger()

    # Seed the file just past the rotation threshold
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x" * (MAX_BYTES + 100))

    log_event("recall", scope="s1", query="will trigger rotation")

    rotated = p.with_suffix(p.suffix + ".1")
    assert rotated.exists(), "old file should have been rotated to .1"
    # New write went to the fresh file
    assert p.exists()
    new_content = p.read_text().strip()
    assert new_content
    rec = json.loads(new_content)
    assert rec["event"] == "recall"


def test_default_path_honors_env(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom" / "events.jsonl"
    monkeypatch.setenv("SKEIN_EVENTS_PATH", str(custom))
    assert default_path() == custom


def test_default_path_fallback(monkeypatch) -> None:
    # Iter 27 Windows port: the per-user state dir lives at
    # %APPDATA%\skein\ on Windows and ~/.config/skein/ on macOS/Linux.
    # Assert against `paths.skein_home()` so the test follows whichever
    # platform it runs on.
    from skein import paths as skein_paths
    monkeypatch.delenv("SKEIN_EVENTS_PATH", raising=False)
    p = default_path()
    assert p.name == "events.jsonl"
    assert p.parent == skein_paths.skein_home()


def test_logger_omits_scope_when_none(tmp_events: Path) -> None:
    log_event("release_lease", lease_id="abc123")
    rec = json.loads(tmp_events.read_text().strip())
    assert "scope" not in rec  # omitted when not given
    assert rec["details"]["lease_id"] == "abc123"
