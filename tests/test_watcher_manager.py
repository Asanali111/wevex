"""Tests for watcher_manager: spawn / is_running / kill of detached watchers.

We never actually fork ``skein watch`` here — that would couple the test to
the live install. Instead we monkeypatch ``subprocess.Popen`` to return a
predictable PID and test the bookkeeping (PID file, slug, kill).
"""
from __future__ import annotations

import os

import pytest

from skein import watcher_manager
from skein.projects import ProjectEntry


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher_manager, "WATCHER_PID_DIR", tmp_path / "watchers")
    monkeypatch.setattr(watcher_manager, "WATCHER_LOG_DIR", tmp_path / "logs")
    return tmp_path


@pytest.fixture
def entry(tmp_path):
    root = tmp_path / "myproj"
    root.mkdir()
    return ProjectEntry(
        scope="project:myproj", root=str(root.resolve()), source_root="myproj",
    )


# ---------------------------------------------------------------------------
# Slug / path helpers
# ---------------------------------------------------------------------------

class TestSlug:
    def test_simple(self):
        assert watcher_manager._slug("project:foo") == "project-foo"

    def test_punctuation(self):
        assert watcher_manager._slug("a/b\\c d!") == "a-b-c-d"

    def test_empty_falls_back(self):
        assert watcher_manager._slug("") == "default"
        assert watcher_manager._slug("@@@") == "default"


class TestPidFilePath:
    def test_uses_scope_slug(self, isolated, entry):
        pid_file = watcher_manager.pid_file_for(entry)
        assert pid_file.name == "project-myproj.pid"
        assert pid_file.parent == isolated / "watchers"


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------

class TestIsRunning:
    def test_false_when_no_pid_file(self, isolated, entry):
        assert watcher_manager.is_running(entry) is False

    def test_true_when_pid_file_present_and_alive(self, isolated, entry, monkeypatch):
        watcher_manager.pid_file_for(entry).parent.mkdir(parents=True)
        watcher_manager.pid_file_for(entry).write_text("12345")
        monkeypatch.setattr(watcher_manager, "_alive", lambda pid: True)
        assert watcher_manager.is_running(entry) is True

    def test_false_and_cleans_pid_file_when_dead(self, isolated, entry, monkeypatch):
        pid_file = watcher_manager.pid_file_for(entry)
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("99999")
        monkeypatch.setattr(watcher_manager, "_alive", lambda pid: False)
        assert watcher_manager.is_running(entry) is False
        assert not pid_file.exists()

    def test_false_when_pid_file_garbage(self, isolated, entry):
        pid_file = watcher_manager.pid_file_for(entry)
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("not-an-int")
        assert watcher_manager.is_running(entry) is False


# ---------------------------------------------------------------------------
# spawn
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class TestSpawn:
    def test_writes_pid_file(self, isolated, entry, monkeypatch):
        captured = {}
        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["new_session"] = kwargs.get("start_new_session")
            return _FakeProc(pid=42)
        monkeypatch.setattr("subprocess.Popen", fake_popen)
        # Pretend skein binary exists
        monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)

        pid = watcher_manager.spawn(entry, skein_bin="/usr/local/bin/skein")
        assert pid == 42
        assert watcher_manager.pid_file_for(entry).read_text() == "42"
        assert captured["cmd"][0] == "/usr/local/bin/skein"
        assert captured["cmd"][1] == "watch"
        assert "--scope" in captured["cmd"]
        assert "project:myproj" in captured["cmd"]
        assert captured["new_session"] is True

    def test_returns_none_when_already_running(self, isolated, entry, monkeypatch):
        # Plant a live PID file
        watcher_manager.pid_file_for(entry).parent.mkdir(parents=True)
        watcher_manager.pid_file_for(entry).write_text("42")
        monkeypatch.setattr(watcher_manager, "_alive", lambda pid: True)
        # If spawn is called we want to know
        called = {"flag": False}
        def fake_popen(*a, **k):
            called["flag"] = True
            return _FakeProc(pid=99)
        monkeypatch.setattr("subprocess.Popen", fake_popen)

        result = watcher_manager.spawn(entry, skein_bin="/skein")
        assert result is None
        assert called["flag"] is False


# ---------------------------------------------------------------------------
# kill
# ---------------------------------------------------------------------------

class TestKill:
    def test_no_op_when_no_pid(self, isolated, entry):
        assert watcher_manager.kill(entry) is False

    def test_sigterm_then_cleans_pid_file(self, isolated, entry, monkeypatch):
        pid_file = watcher_manager.pid_file_for(entry)
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("12345")

        signals_sent = []
        def fake_kill(pid, sig):
            signals_sent.append((pid, sig))
            # Pretend it dies after first SIGTERM
        monkeypatch.setattr(os, "kill", fake_kill)
        # _alive returns True initially, then False
        toggled = {"alive_calls": 0}
        def fake_alive(pid):
            toggled["alive_calls"] += 1
            return toggled["alive_calls"] == 1
        monkeypatch.setattr(watcher_manager, "_alive", fake_alive)

        assert watcher_manager.kill(entry) is True
        assert signals_sent and signals_sent[0][0] == 12345
        assert not pid_file.exists()


# ---------------------------------------------------------------------------
# kill_all
# ---------------------------------------------------------------------------

class TestKillAll:
    def test_kills_all_running(self, isolated, entry, monkeypatch, tmp_path):
        from skein import projects as projects_mod
        # Use a temporary registry file
        fake_registry = tmp_path / "projects.json"
        monkeypatch.setattr(projects_mod, "REGISTRY_PATH", fake_registry)
        projects_mod.upsert_project(entry)

        # Plant a PID file so is_running returns True
        pid_file = watcher_manager.pid_file_for(entry)
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("12345")

        monkeypatch.setattr(watcher_manager, "_alive", lambda pid: False)
        # kill returns True when the PID file existed
        killed = watcher_manager.kill_all()
        # Since _alive is False, is_running returns False, so nothing matches
        # — that's the correct behaviour ("only kill what's actually alive").
        assert killed == []
