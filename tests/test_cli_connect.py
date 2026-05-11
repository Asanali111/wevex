"""Tests for ``skein connect`` / ``disconnect`` / ``clients`` CLI commands."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skein import clients as clients_mod
from skein import connections as conns
from skein.cli import main


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect home, registry, and PATH so the CLI doesn't see real installs."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", tmp_path / "connections.json")
    return tmp_path


# ---------------------------------------------------------------------------
# `skein clients`
# ---------------------------------------------------------------------------

class TestClientsCommand:
    def test_runs_clean_machine(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["clients"])
        assert result.exit_code == 0
        # Every supported id should appear
        for cid in clients_mod.all_ids():
            assert cid in result.output

    def test_json_output(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["clients", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        ids = {d["id"] for d in data}
        for cid in clients_mod.all_ids():
            assert cid in ids
        for entry in data:
            assert "connected" in entry

    def test_shows_connected(self, isolated):
        # Pretend cursor is detected and connected
        (isolated / ".cursor").mkdir()
        conns.mark_connected("cursor", ["/tmp/cursor/mcp.json"])
        runner = CliRunner()
        result = runner.invoke(main, ["clients", "--json"])
        data = json.loads(result.output)
        cursor = next(d for d in data if d["id"] == "cursor")
        assert cursor["connected"] is True
        assert cursor["detected"] is True


# ---------------------------------------------------------------------------
# `skein connect <id>`
# ---------------------------------------------------------------------------

class TestConnectByID:
    def test_unknown_id_errors(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["connect", "does-not-exist"])
        assert result.exit_code != 0
        assert "Unknown client id" in result.output

    def test_not_installed_errors(self, isolated):
        # Cursor not installed (no ~/.cursor and no binary)
        runner = CliRunner()
        result = runner.invoke(main, ["connect", "cursor"])
        assert result.exit_code != 0
        assert "does not appear to be installed" in result.output


class TestConnectAll:
    def test_no_detected_clients_errors(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["connect", "--all"])
        assert result.exit_code != 0
        assert "No supported clients detected" in result.output


# ---------------------------------------------------------------------------
# `skein disconnect`
# ---------------------------------------------------------------------------

class TestDisconnect:
    def test_no_args_errors(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["disconnect"])
        assert result.exit_code != 0
        assert "Pass a client id" in result.output

    def test_unknown_id_errors(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["disconnect", "does-not-exist"])
        assert result.exit_code != 0
        assert "Unknown client id" in result.output

    def test_not_connected_is_noop(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["disconnect", "cursor"])
        assert result.exit_code == 0
        assert "not currently connected" in result.output

    def test_disconnect_all_empty_registry(self, isolated):
        runner = CliRunner()
        result = runner.invoke(main, ["disconnect", "--all"])
        assert result.exit_code == 0
        assert "No clients are currently connected" in result.output

    def test_disconnect_removes_skein_from_config(self, isolated):
        # Pretend cursor is connected with a real config file
        cfg = isolated / ".cursor" / "mcp.json"
        cfg.parent.mkdir()
        cfg.write_text(json.dumps({
            "mcpServers": {
                "skein": {"url": "http://x/mcp"},
                "other": {"url": "http://other"},
            }
        }))
        conns.mark_connected("cursor", [str(cfg)])

        runner = CliRunner()
        result = runner.invoke(main, ["disconnect", "cursor"])
        assert result.exit_code == 0

        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcpServers"]
        assert "other" in data["mcpServers"]
        assert not conns.is_connected("cursor")
