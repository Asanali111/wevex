"""Tests for the Antigravity client adapter.

Antigravity stores its MCP config at ``~/.gemini/antigravity/mcp_config.json``.
We test by monkeypatching ``Path.home()`` so we don't touch the user's real
config.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skein.clients import AntigravityClient


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() so writers operate inside tmp_path. Also blank
    PATH so binary detection doesn't pick up real installs on the test host."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("PATH", "")
    return tmp_path


def _connect(client: AntigravityClient, token: str = "tok") -> None:
    client.connect(
        mcp_url="http://127.0.0.1:8765/mcp",
        bearer_token=token,
        scope_handle="project:test",
        repo=Path.cwd(),
    )


class TestAntigravityClient:
    def test_detect_false_when_not_installed(self, fake_home):
        # No ~/.gemini/antigravity directory and no antigravity binary
        ok, _ = AntigravityClient().detect()
        assert ok is False

    def test_detect_true_when_dir_exists(self, fake_home):
        (fake_home / ".gemini" / "antigravity").mkdir(parents=True)
        ok, _ = AntigravityClient().detect()
        assert ok is True

    def test_writes_correct_path_and_format(self, fake_home):
        ag_dir = fake_home / ".gemini" / "antigravity"
        ag_dir.mkdir(parents=True)
        (ag_dir / "installation_id").write_text("test-id")

        _connect(AntigravityClient(), token="secret-token")

        config = ag_dir / "mcp_config.json"
        assert config.exists()
        data = json.loads(config.read_text())
        assert "skein" in data["mcpServers"]
        skein = data["mcpServers"]["skein"]
        assert skein["serverUrl"] == "http://127.0.0.1:8765/mcp"
        assert skein["headers"]["Authorization"] == "Bearer secret-token"

    def test_preserves_other_servers(self, fake_home):
        ag_dir = fake_home / ".gemini" / "antigravity"
        ag_dir.mkdir(parents=True)
        existing = {
            "mcpServers": {
                "supabase": {"serverUrl": "https://mcp.supabase.com/mcp"},
                "github-mcp": {
                    "command": "docker",
                    "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
                },
            }
        }
        (ag_dir / "mcp_config.json").write_text(json.dumps(existing))

        _connect(AntigravityClient())

        data = json.loads((ag_dir / "mcp_config.json").read_text())
        assert "supabase" in data["mcpServers"]
        assert "github-mcp" in data["mcpServers"]
        assert "skein" in data["mcpServers"]

    def test_cleans_up_legacy_company_brain_entry(self, fake_home):
        ag_dir = fake_home / ".gemini" / "antigravity"
        ag_dir.mkdir(parents=True)
        existing = {
            "mcpServers": {
                "company-brain": {
                    "command": "brain",
                    "args": ["mcp"],
                    "env": {"MCP_TEAM_ID": "stale"},
                },
                "supabase": {"serverUrl": "https://mcp.supabase.com/mcp"},
            }
        }
        (ag_dir / "mcp_config.json").write_text(json.dumps(existing))

        _connect(AntigravityClient())

        data = json.loads((ag_dir / "mcp_config.json").read_text())
        assert "company-brain" not in data["mcpServers"]
        assert "supabase" in data["mcpServers"]
        assert "skein" in data["mcpServers"]

    def test_idempotent(self, fake_home):
        ag_dir = fake_home / ".gemini" / "antigravity"
        ag_dir.mkdir(parents=True)

        client = AntigravityClient()
        for i in range(3):
            _connect(client, token=f"tok-{i}")

        data = json.loads((ag_dir / "mcp_config.json").read_text())
        assert list(data["mcpServers"]).count("skein") == 1
        assert data["mcpServers"]["skein"]["headers"]["Authorization"] == "Bearer tok-2"

    def test_unreadable_file_backed_up(self, fake_home):
        ag_dir = fake_home / ".gemini" / "antigravity"
        ag_dir.mkdir(parents=True)
        (ag_dir / "mcp_config.json").write_text("{this is not json")

        _connect(AntigravityClient())

        # Backup created
        assert (ag_dir / "mcp_config.json.bak").exists()
        # New file is valid JSON with our entry
        data = json.loads((ag_dir / "mcp_config.json").read_text())
        assert "skein" in data["mcpServers"]
