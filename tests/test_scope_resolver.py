"""Tests for the scope resolution helper used by both CLI and hook handlers."""
from __future__ import annotations

from pathlib import Path

import pytest

from skein.scope_resolver import find_scope_pin, resolve_scope


class TestFindScopePin:
    def test_pin_in_cwd(self, tmp_path, monkeypatch):
        skein_dir = tmp_path / ".skein"
        skein_dir.mkdir()
        (skein_dir / "scope").write_text("project:foo\n")
        monkeypatch.chdir(tmp_path)
        assert find_scope_pin() == "project:foo"

    def test_pin_in_parent(self, tmp_path, monkeypatch):
        # Pin lives at the repo root; we run from a deep subdirectory
        skein_dir = tmp_path / ".skein"
        skein_dir.mkdir()
        (skein_dir / "scope").write_text("project:repo\n")

        deep = tmp_path / "src" / "lib" / "auth"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        assert find_scope_pin() == "project:repo"

    def test_no_pin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert find_scope_pin() is None

    def test_empty_pin_treated_as_missing(self, tmp_path, monkeypatch):
        skein_dir = tmp_path / ".skein"
        skein_dir.mkdir()
        (skein_dir / "scope").write_text("   \n")
        monkeypatch.chdir(tmp_path)
        assert find_scope_pin() is None


class TestResolveScope:
    def test_cli_arg_wins(self, tmp_path, monkeypatch):
        # Even with env + pin set, --scope wins
        skein_dir = tmp_path / ".skein"
        skein_dir.mkdir()
        (skein_dir / "scope").write_text("project:from-pin")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SKEIN_SCOPE", "project:from-env")

        scope, source = resolve_scope(
            "project:from-cli",
            config_default="project:from-config",
        )
        assert scope == "project:from-cli"
        assert source == "cli"

    def test_env_beats_pin(self, tmp_path, monkeypatch):
        skein_dir = tmp_path / ".skein"
        skein_dir.mkdir()
        (skein_dir / "scope").write_text("project:from-pin")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SKEIN_SCOPE", "project:from-env")

        scope, source = resolve_scope(None, config_default="project:from-config")
        assert scope == "project:from-env"
        assert source == "env"

    def test_pin_beats_config(self, tmp_path, monkeypatch):
        skein_dir = tmp_path / ".skein"
        skein_dir.mkdir()
        (skein_dir / "scope").write_text("project:from-pin")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SKEIN_SCOPE", raising=False)

        scope, source = resolve_scope(None, config_default="project:from-config")
        assert scope == "project:from-pin"
        assert source == "pin"

    def test_config_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SKEIN_SCOPE", raising=False)
        scope, source = resolve_scope(None, config_default="project:from-config")
        assert scope == "project:from-config"
        assert source == "config"

    def test_no_default_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SKEIN_SCOPE", raising=False)
        with pytest.raises(RuntimeError):
            resolve_scope(None, config_default=None)
