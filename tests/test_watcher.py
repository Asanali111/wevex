"""Tests for the file watcher's re-ingest dispatcher.

We test the polling backend directly (deterministic, no fsevents timing).
The watchdog backend uses the same _BaseWatcher logic, so behavioural tests
on _PollingWatcher cover both.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from skein.embeddings import HashEmbeddingProvider
from skein.models import IdentityCreate, ScopeCreate
from skein.watcher import _PollingWatcher, _WatchdogWatcher, make_watcher


@pytest.fixture
def watch_setup(seeded_storage, tmp_path):
    """Storage + scope + project root with one tracked file."""
    s = seeded_storage
    scope = s._test_scope
    project_root = tmp_path / "watched_proj"
    (project_root / "src").mkdir(parents=True)
    (project_root / "src" / "main.py").write_text("def hello():\n    return 'world'\n")
    return {
        "storage": s,
        "scope_id": scope.id,
        "root": project_root,
        "provider": HashEmbeddingProvider(),
    }


# ---------------------------------------------------------------------------
# _BaseWatcher: filtering
# ---------------------------------------------------------------------------

class TestShouldTrack:
    def test_includes_python(self, watch_setup):
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
        )
        assert w._should_track(watch_setup["root"] / "src" / "main.py")

    def test_excludes_node_modules(self, watch_setup):
        nm = watch_setup["root"] / "node_modules" / "x.js"
        nm.parent.mkdir(parents=True)
        nm.write_text("//")
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
        )
        assert not w._should_track(nm)

    def test_excludes_unknown_extension(self, watch_setup):
        bin_path = watch_setup["root"] / "blob.bin"
        bin_path.write_bytes(b"\x00\x01\x02")
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
        )
        assert not w._should_track(bin_path)

    def test_rejects_paths_outside_root(self, watch_setup, tmp_path):
        outside = tmp_path / "elsewhere.py"
        outside.write_text("# nope")
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
        )
        assert not w._should_track(outside)


# ---------------------------------------------------------------------------
# _BaseWatcher._reingest: end-to-end without observer threads
# ---------------------------------------------------------------------------

class TestReingestDispatch:
    def test_creates_chunks_for_new_file(self, watch_setup):
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
            debounce_secs=0,   # immediate flush
        )
        target = watch_setup["root"] / "src" / "main.py"
        w._reingest(target)
        assert w.stats.files_reingested == 1

        chunks = watch_setup["storage"].list_chunks(
            scope_id=watch_setup["scope_id"], limit=10,
        )
        assert any(c.source_path == "src/main.py" for c in chunks)

    def test_updates_chunks_on_content_change(self, watch_setup):
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
            debounce_secs=0,
        )
        target = watch_setup["root"] / "src" / "main.py"
        w._reingest(target)

        # Edit and re-ingest
        target.write_text(
            "def hello():\n    return 'changed!'\n\ndef extra():\n    pass\n"
        )
        w._reingest(target)

        chunks = watch_setup["storage"].list_chunks(
            scope_id=watch_setup["scope_id"], limit=10,
        )
        contents = [c.content for c in chunks if c.source_path == "src/main.py"]
        assert any("changed" in c for c in contents)
        assert not any("'world'" in c for c in contents)

    def test_prunes_chunks_for_shrinking_file(self, watch_setup):
        # Create a long file that produces multiple chunks
        big = "\n".join([f"line {i}" for i in range(200)])
        target = watch_setup["root"] / "big.py"
        target.write_text(big)

        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
            debounce_secs=0, chunk_lines=80, overlap_lines=10,
        )
        w._reingest(target)
        before = [c for c in watch_setup["storage"].list_chunks(
            scope_id=watch_setup["scope_id"], limit=200,
        ) if c.source_path == "big.py"]
        assert len(before) >= 2

        # Replace with a much shorter file
        target.write_text("just one line\n")
        w._reingest(target)
        after = [c for c in watch_setup["storage"].list_chunks(
            scope_id=watch_setup["scope_id"], limit=200,
        ) if c.source_path == "big.py"]
        assert len(after) == 1
        assert "just one line" in after[0].content

    def test_deletion_removes_chunks(self, watch_setup):
        target = watch_setup["root"] / "src" / "main.py"
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
            debounce_secs=0,
        )
        w._reingest(target)
        assert any(c.source_path == "src/main.py" for c in
                   watch_setup["storage"].list_chunks(
                       scope_id=watch_setup["scope_id"], limit=10,
                   ))

        target.unlink()
        w._enqueue_deleted(target)

        remaining = [c for c in watch_setup["storage"].list_chunks(
            scope_id=watch_setup["scope_id"], limit=10,
        ) if c.source_path == "src/main.py"]
        assert remaining == []

    def test_skips_oversized_files(self, watch_setup):
        big = watch_setup["root"] / "big.py"
        big.write_text("x = 1\n" * 200_000)  # ~1.2 MB

        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
            debounce_secs=0, max_file_bytes=100_000,
        )
        w._reingest(big)
        assert w.stats.files_reingested == 0

    def test_skips_binary_files(self, watch_setup):
        # Wrong extension also skipped, but verify decode failure path too
        path = watch_setup["root"] / "weird.py"
        path.write_bytes(b"\xff\xfe\x00\x01\x02\x03")
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
            debounce_secs=0,
        )
        w._reingest(path)
        assert w.stats.files_reingested == 0


# ---------------------------------------------------------------------------
# Polling backend end-to-end (with timing)
# ---------------------------------------------------------------------------

class TestPollingWatcher:
    def test_detects_new_file_via_loop(self, watch_setup):
        w = _PollingWatcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
            debounce_secs=0.1, poll_interval=0.2,
        )
        w.start()
        try:
            new_file = watch_setup["root"] / "src" / "added.py"
            new_file.write_text("def added():\n    pass\n")
            # Wait up to 5s for the poll loop to see it
            deadline = time.time() + 5
            while time.time() < deadline:
                chunks = watch_setup["storage"].list_chunks(
                    scope_id=watch_setup["scope_id"], limit=50,
                )
                if any(c.source_path == "src/added.py" for c in chunks):
                    break
                time.sleep(0.2)
            chunks = watch_setup["storage"].list_chunks(
                scope_id=watch_setup["scope_id"], limit=50,
            )
            assert any(c.source_path == "src/added.py" for c in chunks)
        finally:
            w.stop()


# ---------------------------------------------------------------------------
# make_watcher factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_picks_watchdog_when_available(self, watch_setup, monkeypatch):
        # Don't actually start; just check the type
        from skein import watcher as wmod
        monkeypatch.setattr(wmod, "_watchdog_available", lambda: True)
        w = make_watcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
        )
        try:
            assert isinstance(w, _WatchdogWatcher)
        finally:
            w.stop()

    def test_falls_back_to_polling(self, watch_setup, monkeypatch):
        from skein import watcher as wmod
        monkeypatch.setattr(wmod, "_watchdog_available", lambda: False)
        w = make_watcher(
            watch_setup["root"], watch_setup["scope_id"], "watched_proj",
            watch_setup["storage"], watch_setup["provider"],
        )
        try:
            assert isinstance(w, _PollingWatcher)
        finally:
            w.stop()
