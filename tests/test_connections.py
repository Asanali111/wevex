"""Tests for ``skein.connections`` registry — CRUD on connections.json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skein import connections as conns


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    path = tmp_path / "connections.json"
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", path)
    return path


class TestEmptyRegistry:
    def test_get_connected_ids_empty(self, fake_registry):
        assert conns.get_connected_ids() == []

    def test_is_connected_false(self, fake_registry):
        assert conns.is_connected("cursor") is False

    def test_get_connection_none(self, fake_registry):
        assert conns.get_connection("cursor") is None

    def test_list_all_empty(self, fake_registry):
        assert conns.list_all() == {}

    def test_get_paths_empty(self, fake_registry):
        assert conns.get_paths("cursor") == []

    def test_disconnect_missing_returns_false(self, fake_registry):
        assert conns.mark_disconnected("cursor") is False


class TestMarkConnected:
    def test_creates_file(self, fake_registry):
        conns.mark_connected("cursor", ["/tmp/cursor/mcp.json"])
        assert fake_registry.exists()
        data = json.loads(fake_registry.read_text())
        assert "cursor" in data
        assert data["cursor"]["config_paths"] == ["/tmp/cursor/mcp.json"]
        assert data["cursor"]["connected_at"].endswith("Z")

    def test_overwrites_existing(self, fake_registry):
        conns.mark_connected("cursor", ["/tmp/a"])
        conns.mark_connected("cursor", ["/tmp/b"])
        assert conns.get_paths("cursor") == ["/tmp/b"]

    def test_multiple_clients(self, fake_registry):
        conns.mark_connected("cursor", ["/tmp/cursor"])
        conns.mark_connected("gemini_cli", ["/tmp/gemini"])
        ids = conns.get_connected_ids()
        assert ids == ["cursor", "gemini_cli"]
        assert conns.is_connected("cursor")
        assert conns.is_connected("gemini_cli")
        assert not conns.is_connected("vscode")

    def test_paths_serialised_as_strings(self, fake_registry):
        # Iter 27 Windows port: passing a `Path` here was being normalized
        # to backslashes on Windows (`\tmp\cursor\mcp.json`). The test's
        # point is "Path objects become strings in JSON"; build the input
        # via Path() but assert against the OS-native string form so the
        # test is platform-independent.
        p = Path("/tmp/cursor/mcp.json")
        conns.mark_connected("cursor", [p])
        data = json.loads(fake_registry.read_text())
        assert data["cursor"]["config_paths"] == [str(p)]


class TestMarkDisconnected:
    def test_removes_entry(self, fake_registry):
        conns.mark_connected("cursor", ["/tmp/cursor"])
        conns.mark_connected("vscode", ["/tmp/vscode"])
        assert conns.mark_disconnected("cursor") is True
        assert conns.get_connected_ids() == ["vscode"]

    def test_persists_to_disk(self, fake_registry):
        conns.mark_connected("cursor", ["/tmp/cursor"])
        conns.mark_disconnected("cursor")
        data = json.loads(fake_registry.read_text())
        assert "cursor" not in data


class TestCorruptedRegistry:
    def test_returns_empty_dict_on_garbage(self, fake_registry):
        fake_registry.write_text("{not valid json")
        assert conns.list_all() == {}

    def test_can_recover_after_corruption(self, fake_registry):
        fake_registry.write_text("not json")
        # mark_connected should overwrite cleanly
        conns.mark_connected("cursor", ["/tmp/cursor"])
        assert conns.is_connected("cursor")
