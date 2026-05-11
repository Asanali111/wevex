"""Tests for auto_detect_scope() — the heuristic used by `skein up`."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from skein.scope_resolver import auto_detect_scope, _clean_handle_part


class TestCleanHandlePart:
    def test_lowercases(self):
        assert _clean_handle_part("MyApp") == "myapp"

    def test_replaces_spaces_and_punctuation(self):
        assert _clean_handle_part("My Cool App!") == "my-cool-app"

    def test_strips_dashes(self):
        assert _clean_handle_part("---weird---") == "weird"

    def test_empty_falls_back_to_default(self):
        assert _clean_handle_part("") == "default"
        assert _clean_handle_part("@@@") == "default"


class TestAutoDetect:
    def test_pin_wins(self, tmp_path, monkeypatch):
        skein_dir = tmp_path / ".skein"
        skein_dir.mkdir()
        (skein_dir / "scope").write_text("project:from-pin")
        monkeypatch.chdir(tmp_path)
        assert auto_detect_scope() == "project:from-pin"

    def test_falls_back_to_dirname(self, tmp_path, monkeypatch):
        target = tmp_path / "MyCoolApp"
        target.mkdir()
        monkeypatch.chdir(target)
        # Not a git repo and no pin
        assert auto_detect_scope() == "project:mycoolapp"

    def test_uses_git_remote_basename(self, tmp_path, monkeypatch):
        # Create a fake git repo with a remote
        repo = tmp_path / "anything"
        repo.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/foo/cool-thing.git"],
            cwd=repo, check=True,
        )
        monkeypatch.chdir(repo)
        assert auto_detect_scope() == "project:cool-thing"

    def test_handles_ssh_remote(self, tmp_path, monkeypatch):
        repo = tmp_path / "ssh-repo"
        repo.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:foo/secret-app.git"],
            cwd=repo, check=True,
        )
        monkeypatch.chdir(repo)
        assert auto_detect_scope() == "project:secret-app"

    def test_handles_git_remote_without_dot_git(self, tmp_path, monkeypatch):
        repo = tmp_path / "no-dot-git"
        repo.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://gitlab.com/bar/widgets"],
            cwd=repo, check=True,
        )
        monkeypatch.chdir(repo)
        assert auto_detect_scope() == "project:widgets"
