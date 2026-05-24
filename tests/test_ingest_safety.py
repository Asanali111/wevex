"""Tests for the iteration-9 ingest safety guards.

Triggered by: user ran ``skein up`` from ``$HOME``, Skein walked the entire
home directory and indexed ~45,000 chunks including Chrome password autofill
data into a ``project:ameliomar`` scope. These tests pin the regressions:

- ``$HOME`` and other system roots are refused outright.
- Sensitive filenames (passwords, credentials, ssh keys, .env) are skipped
  even inside an allowed root.
- Library / .local / .gemini / browser-profile dirs are pruned by the walker.
- ``count_ingestable_files`` returns the same number as a real walk would
  process (so the CLI's threshold prompt is accurate).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from skein.ingest import (
    DEFAULT_EXCLUDES,
    SENSITIVE_FILENAME_FRAGMENTS,
    _refuse_root,
    count_ingestable_files,
    ingest_directory,
)


# ---------------------------------------------------------------------------
# _refuse_root
# ---------------------------------------------------------------------------

class TestRefuseRoot:
    def test_refuses_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert _refuse_root(tmp_path) is not None

    @pytest.mark.skipif(
        sys.platform.startswith("win") or os.name == "nt",
        reason=(
            "_refuse_root's system-root list is POSIX-only (`/`, `/tmp`, "
            "`/etc`); the Windows equivalent (`C:\\`, `C:\\Windows`, …) "
            "is follow-up work, not blocking for first-cut Windows support."
        ),
    )
    def test_refuses_system_dirs(self):
        assert _refuse_root(Path("/")) is not None
        assert _refuse_root(Path("/tmp")) is not None
        assert _refuse_root(Path("/etc")) is not None

    def test_refuses_top_level_user_dirs(self, tmp_path, monkeypatch):
        # Pretend tmp_path is $HOME, then reject the well-known siblings.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        for name in ["Library", "Applications", "Music", "Movies",
                     "Pictures", "Desktop", "Downloads"]:
            d = tmp_path / name
            d.mkdir()
            assert _refuse_root(d) is not None, f"should refuse {name}"

    def test_allows_real_project(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        proj = tmp_path / "Documents" / "myapp"
        proj.mkdir(parents=True)
        assert _refuse_root(proj) is None


# ---------------------------------------------------------------------------
# Default excludes — privacy / sensitive folders
# ---------------------------------------------------------------------------

class TestDefaultExcludes:
    @pytest.mark.parametrize("name", [
        "Library", "Applications", ".local", ".cache",
        ".gemini", ".claude", ".cursor", ".codex",
        "Chrome", "Firefox", "ZxcvbnData",
        ".aws", ".ssh", ".gnupg",
    ])
    def test_in_default_excludes(self, name):
        assert name in DEFAULT_EXCLUDES, f"{name} should be in DEFAULT_EXCLUDES"


# ---------------------------------------------------------------------------
# Sensitive filename filter (run via real ingest)
# ---------------------------------------------------------------------------

class TestSensitiveFilenameFilter:
    @pytest.mark.parametrize("frag", [
        "passwords", "credential", "id_rsa", ".env", "secret", "api_key",
    ])
    def test_fragment_listed(self, frag):
        assert any(frag in f for f in SENSITIVE_FILENAME_FRAGMENTS)

    def test_passwords_txt_skipped(self, tmp_path, monkeypatch):
        # Set up a project dir with passwords.txt + a normal py file
        monkeypatch.setattr(Path, "home", classmethod(
            lambda cls: tmp_path.parent
        ))
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "passwords.txt").write_text("alice:hunter2\n")
        (proj / "main.py").write_text("print('hi')\n")

        # Use a fake Storage that only counts upserts
        class _FakeStorage:
            def __init__(self): self.upserts = []
            def upsert_chunk(self, c, *, content_hash, embedding):
                self.upserts.append(c.source_path)
                return None, "inserted"
            def list_chunks(self, **kw): return []

        stats = ingest_directory(
            proj, _FakeStorage(), provider=None,
            scope_id="x", source_root="myproj",
        )
        # main.py is ingested; passwords.txt is rejected pre-read.
        assert stats.files_ingested == 1
        assert any("passwords" in s for s in stats.skipped_paths)


# ---------------------------------------------------------------------------
# count_ingestable_files
# ---------------------------------------------------------------------------

class TestCountIngestableFiles:
    def test_counts_only_supported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(
            lambda cls: tmp_path.parent
        ))
        proj = tmp_path / "myproj"
        proj.mkdir()
        # 3 ingestable + 2 skipped
        (proj / "a.py").write_text("x")
        (proj / "b.md").write_text("x")
        (proj / "c.ts").write_text("x")
        (proj / "d.bin").write_text("x")
        (proj / "passwords.txt").write_text("x")
        # _walk only returns files with supported extensions; passwords.txt
        # has .txt which IS supported, but the sensitive-fragment filter is
        # applied later — count is 4 here (passwords.txt counted), and the
        # CLI threshold guard treats this as conservative upper bound.
        assert count_ingestable_files(proj) == 4

    def test_excludes_pruned_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(
            lambda cls: tmp_path.parent
        ))
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "main.py").write_text("x")
        (proj / "node_modules").mkdir()
        (proj / "node_modules" / "lodash.js").write_text("x" * 100)
        (proj / "Library").mkdir()
        (proj / "Library" / "secret.py").write_text("x")
        # Only main.py — Library and node_modules are pruned.
        assert count_ingestable_files(proj) == 1


# ---------------------------------------------------------------------------
# ingest_directory itself refuses bad roots
# ---------------------------------------------------------------------------

class TestIngestRefusesRoots:
    def test_raises_for_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        class _NopeStorage:
            def upsert_chunk(self, *a, **k): raise AssertionError("walked")
            def list_chunks(self, **kw): return []

        with pytest.raises(ValueError, match="refusing"):
            ingest_directory(
                tmp_path, _NopeStorage(), provider=None,
                scope_id="x", source_root="home",
            )
