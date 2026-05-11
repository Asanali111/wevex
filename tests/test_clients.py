"""Tests for ``skein.clients`` — detection, connect, disconnect per client."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skein import clients as clients_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() and clear PATH so detection doesn't match the
    test host's real installs."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("PATH", "")
    return tmp_path


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    return r


# ---------------------------------------------------------------------------
# detect_all
# ---------------------------------------------------------------------------

class TestDetectAll:
    def test_returns_entry_for_every_known_client(self, fake_home):
        out = clients_mod.detect_all()
        ids = {e["id"] for e in out}
        for cid in clients_mod.all_ids():
            assert cid in ids

    def test_clean_machine_detects_nothing(self, fake_home):
        out = clients_mod.detect_all()
        for entry in out:
            assert entry["detected"] is False

    def test_picks_up_dir_signal(self, fake_home):
        (fake_home / ".cursor").mkdir()
        out = {e["id"]: e for e in clients_mod.detect_all()}
        assert out["cursor"]["detected"] is True

    def test_each_entry_has_required_keys(self, fake_home):
        for entry in clients_mod.detect_all():
            assert {"id", "display_name", "description", "detected", "note"} <= set(entry.keys())


# ---------------------------------------------------------------------------
# get_client / all_ids
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_ids_unique(self):
        ids = clients_mod.all_ids()
        assert len(ids) == len(set(ids))

    def test_get_client_known(self):
        c = clients_mod.get_client("cursor")
        assert c is not None and c.id == "cursor"

    def test_get_client_unknown(self):
        assert clients_mod.get_client("does-not-exist") is None


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

class TestCursorClient:
    def test_connect_writes_mcp_json(self, fake_home, repo):
        client = clients_mod.CursorClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        cfg = repo / ".cursor" / "mcp.json"
        assert cfg.exists()
        assert str(cfg) in paths
        data = json.loads(cfg.read_text())
        assert data["mcpServers"]["skein"]["url"] == "http://x/mcp"
        assert data["mcpServers"]["skein"]["headers"]["Authorization"] == "Bearer tok"

    def test_connect_preserves_other_servers(self, fake_home, repo):
        cfg = repo / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"mcpServers": {"other": {"url": "http://other"}}}))
        clients_mod.CursorClient().connect("http://x/mcp", "tok", "p", repo)
        data = json.loads(cfg.read_text())
        assert "other" in data["mcpServers"]
        assert "skein" in data["mcpServers"]

    def test_disconnect_removes_skein_only(self, fake_home, repo):
        cfg = repo / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({
            "mcpServers": {
                "skein": {"url": "http://x/mcp"},
                "other": {"url": "http://other"},
            }
        }))
        modified = clients_mod.CursorClient().disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcpServers"]
        assert "other" in data["mcpServers"]
        assert str(cfg) in modified


# ---------------------------------------------------------------------------
# VS Code
# ---------------------------------------------------------------------------

class TestVsCodeClient:
    def test_connect_writes_to_vscode_dir(self, fake_home, repo):
        clients_mod.VsCodeClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = repo / ".vscode" / "mcp.json"
        assert cfg.exists()
        assert json.loads(cfg.read_text())["mcpServers"]["skein"]["url"] == "http://x/mcp"


# ---------------------------------------------------------------------------
# Gemini CLI
# ---------------------------------------------------------------------------

class TestGeminiCLIClient:
    def test_connect_writes_to_home_gemini(self, fake_home, repo):
        clients_mod.GeminiCLIClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = fake_home / ".gemini" / "settings.json"
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert data["mcpServers"]["skein"]["url"] == "http://x/mcp"

    def test_disconnect_removes_skein(self, fake_home, repo):
        client = clients_mod.GeminiCLIClient()
        client.connect("http://x/mcp", "tok", "p", repo)
        cfg = fake_home / ".gemini" / "settings.json"
        client.disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcpServers"]


# ---------------------------------------------------------------------------
# opencode (nested key chain)
# ---------------------------------------------------------------------------

class TestOpenCodeClient:
    def test_connect_writes_nested(self, fake_home, repo):
        clients_mod.OpenCodeClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = fake_home / ".config" / "opencode" / "config.json"
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert data["mcp"]["servers"]["skein"]["url"] == "http://x/mcp"

    def test_disconnect_removes_nested(self, fake_home, repo):
        client = clients_mod.OpenCodeClient()
        client.connect("http://x/mcp", "tok", "p", repo)
        cfg = fake_home / ".config" / "opencode" / "config.json"
        client.disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcp"]["servers"]


# ---------------------------------------------------------------------------
# Codex (TOML)
# ---------------------------------------------------------------------------

class TestCodexClient:
    def test_connect_appends_block(self, fake_home, repo):
        clients_mod.CodexClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = repo / ".codex" / "config.toml"
        assert cfg.exists()
        text = cfg.read_text()
        assert "[[mcpServers]]" in text
        assert 'name = "skein"' in text
        assert 'url = "http://x/mcp"' in text
        assert 'Authorization = "Bearer tok"' in text

    def test_connect_idempotent(self, fake_home, repo):
        c = clients_mod.CodexClient()
        c.connect("http://x/mcp", "tok", "p", repo)
        c.connect("http://x/mcp", "tok2", "p", repo)
        text = (repo / ".codex" / "config.toml").read_text()
        # second call detects existing skein block and bails
        assert text.count('name = "skein"') == 1

    def test_connect_preserves_user_blocks(self, fake_home, repo):
        cfg = repo / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[user]\nname = "alice"\n\n'
            '[[mcpServers]]\nname = "other"\nurl = "http://other"\n'
        )
        clients_mod.CodexClient().connect("http://x/mcp", "tok", "p", repo)
        text = cfg.read_text()
        assert 'name = "alice"' in text
        assert 'name = "other"' in text
        assert 'name = "skein"' in text

    def test_disconnect_strips_skein_block_only(self, fake_home, repo):
        cfg = repo / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[user]\nname = "alice"\n\n'
            '[[mcpServers]]\nname = "other"\nurl = "http://other"\n\n'
        )
        client = clients_mod.CodexClient()
        client.connect("http://x/mcp", "tok", "p", repo)
        client.disconnect(recorded_paths=[str(cfg)])
        text = cfg.read_text()
        assert 'name = "alice"' in text
        assert 'name = "other"' in text
        assert 'name = "skein"' not in text


# ---------------------------------------------------------------------------
# Disconnect: empty registry, missing files, multiple paths
# ---------------------------------------------------------------------------

class TestDisconnectResilience:
    def test_disconnect_no_recorded_paths_no_files(self, fake_home, repo):
        # Run from inside a tmp working directory so cwd-based fallbacks
        # don't accidentally hit user files.
        modified = clients_mod.CursorClient().disconnect(recorded_paths=[])
        assert modified == [] or modified == []  # always benign

    def test_disconnect_missing_recorded_file(self, fake_home, repo):
        modified = clients_mod.CursorClient().disconnect(
            recorded_paths=[str(repo / "nonexistent")]
        )
        assert modified == []
