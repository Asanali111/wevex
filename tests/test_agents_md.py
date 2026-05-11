"""Tests for agents_md.py and sync.py."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from skein.agents_md import (
    _extract_user_block,
    render_agents_md,
    render_claude_md_shim,
)
from skein.models import FragmentCreate, IdentityCreate, ScopeCreate
from skein.storage import Storage


@pytest.fixture
def populated_storage(storage: Storage) -> Storage:
    user = storage.create_identity(IdentityCreate(
        handle="user:test", type="user", name="Test",
    ))
    scope = storage.create_scope(ScopeCreate(
        handle="project:agents-md-test", type="project",
        name="Agents MD Test", owner_id=user.id,
    ))
    storage._test_user = user
    storage._test_scope = scope

    fragments = [
        ("requirement", "users must authenticate with MFA", None),
        ("requirement", "all API responses must be < 200ms", "backend/api"),
        ("preference", "prefer async/await over callbacks", "backend"),
        ("decision", "use Redis for session caching", "backend/cache"),
        ("decision", "use PostgreSQL as primary database", "backend/db"),
        ("state", "Redis running on port 6379", "backend/cache"),
        ("procedure", "To add a new API endpoint: 1. Define schema 2. Add route 3. Write tests", "backend"),
    ]
    for ftype, content, territory in fragments:
        storage.create_fragment(FragmentCreate(
            type=ftype, content=content,
            scope_id=scope.id, owner_id=user.id,
            territory=territory,
        ))
    return storage


# ---------------------------------------------------------------------------
# render_agents_md
# ---------------------------------------------------------------------------

def test_agents_md_has_header(populated_storage: Storage) -> None:
    content = render_agents_md(
        "project:agents-md-test",
        populated_storage,
        daemon_url="http://127.0.0.1:8765",
    )
    assert "AGENTS.md" in content
    assert "Skein" in content or "skein" in content
    assert "http://127.0.0.1:8765/mcp" in content


def test_agents_md_has_recall_instructions(populated_storage: Storage) -> None:
    content = render_agents_md("project:agents-md-test", populated_storage)
    assert "recall" in content
    assert "remember" in content


def test_agents_md_shows_requirements(populated_storage: Storage) -> None:
    content = render_agents_md("project:agents-md-test", populated_storage)
    assert "Requirements" in content
    assert "MFA" in content
    assert "200ms" in content


def test_agents_md_shows_decisions(populated_storage: Storage) -> None:
    content = render_agents_md("project:agents-md-test", populated_storage)
    assert "decisions" in content.lower() or "Active decisions" in content
    assert "Redis" in content
    assert "PostgreSQL" in content


def test_agents_md_shows_preferences(populated_storage: Storage) -> None:
    content = render_agents_md("project:agents-md-test", populated_storage)
    assert "Preferences" in content
    assert "async" in content


def test_agents_md_shows_state(populated_storage: Storage) -> None:
    content = render_agents_md("project:agents-md-test", populated_storage)
    assert "state" in content.lower() or "Current state" in content
    assert "6379" in content


def test_agents_md_shows_procedures(populated_storage: Storage) -> None:
    content = render_agents_md("project:agents-md-test", populated_storage)
    assert "Procedures" in content
    assert "Define schema" in content


def test_agents_md_has_user_block(populated_storage: Storage) -> None:
    content = render_agents_md("project:agents-md-test", populated_storage)
    assert "<!-- @user -->" in content
    assert "<!-- /@user -->" in content


def test_agents_md_preserves_user_block(populated_storage: Storage) -> None:
    existing = """# Some existing AGENTS.md
<!-- @user -->
My custom instructions here.
Don't delete me!
<!-- /@user -->
"""
    content = render_agents_md(
        "project:agents-md-test",
        populated_storage,
        existing_content=existing,
    )
    assert "My custom instructions here." in content
    assert "Don't delete me!" in content


def test_agents_md_unknown_scope(storage: Storage) -> None:
    content = render_agents_md("project:no-such-scope", storage)
    assert "not found" in content.lower() or "skein init" in content.lower()


# ---------------------------------------------------------------------------
# render_claude_md_shim
# ---------------------------------------------------------------------------

def test_claude_md_shim() -> None:
    content = render_claude_md_shim()
    assert content.strip() == "@AGENTS.md"


# ---------------------------------------------------------------------------
# _extract_user_block
# ---------------------------------------------------------------------------

def test_extract_user_block_present() -> None:
    content = "before\n<!-- @user -->\nhello world\n<!-- /@user -->\nafter"
    block = _extract_user_block(content)
    assert "hello world" in block


def test_extract_user_block_absent() -> None:
    block = _extract_user_block("no user block here")
    assert block == ""


def test_extract_user_block_whitespace_tolerance() -> None:
    content = "<!-- @user -->\n  spaced\n<!-- /@user -->"
    block = _extract_user_block(content)
    assert "spaced" in block


# ---------------------------------------------------------------------------
# sync.py
# ---------------------------------------------------------------------------

def test_sync_writes_cursor_mcp(tmp_path: Path, monkeypatch) -> None:
    from skein.sync import sync_all
    from skein import connections as conns
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", tmp_path / "connections.json")

    result = sync_all(
        daemon_url="http://127.0.0.1:8765",
        bearer_token="testtoken",
        scope_handle="project:test",
        repo_path=tmp_path,
        agents_md_content="# AGENTS.md\nTest content",
        client_ids=["cursor"],
    )

    cursor_config = tmp_path / ".cursor" / "mcp.json"
    assert cursor_config.exists()
    data = json.loads(cursor_config.read_text())
    assert "skein" in data["mcpServers"]
    assert data["mcpServers"]["skein"]["url"] == "http://127.0.0.1:8765/mcp"


def test_sync_writes_vscode_mcp(tmp_path: Path, monkeypatch) -> None:
    from skein.sync import sync_all
    from skein import connections as conns
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", tmp_path / "connections.json")

    sync_all(
        daemon_url="http://127.0.0.1:8765",
        bearer_token="testtoken",
        scope_handle="project:test",
        repo_path=tmp_path,
        client_ids=["vscode"],
    )
    vscode = tmp_path / ".vscode" / "mcp.json"
    assert vscode.exists()
    data = json.loads(vscode.read_text())
    assert "skein" in data["mcpServers"]


def test_sync_writes_agents_md(tmp_path: Path, monkeypatch) -> None:
    from skein.sync import sync_all
    from skein import connections as conns
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", tmp_path / "connections.json")

    agents_md = "# AGENTS.md\nTest content for agents."
    sync_all(
        daemon_url="http://127.0.0.1:8765",
        bearer_token="testtoken",
        scope_handle="project:test",
        repo_path=tmp_path,
        agents_md_content=agents_md,
        client_ids=[],
    )
    path = tmp_path / "AGENTS.md"
    assert path.exists()
    assert "Test content for agents." in path.read_text()


def test_sync_writes_claude_md_shim(tmp_path: Path, monkeypatch) -> None:
    from skein.sync import sync_all
    from skein import connections as conns
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", tmp_path / "connections.json")

    sync_all(
        daemon_url="http://127.0.0.1:8765",
        bearer_token="testtoken",
        scope_handle="project:test",
        repo_path=tmp_path,
        client_ids=[],
    )
    path = tmp_path / "CLAUDE.md"
    assert path.exists()
    assert path.read_text().strip() == "@AGENTS.md"


def test_sync_does_not_overwrite_custom_claude_md(tmp_path: Path, monkeypatch) -> None:
    from skein.sync import sync_all
    from skein import connections as conns
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", tmp_path / "connections.json")

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# My custom CLAUDE.md\n\nDon't touch this.\n")

    sync_all(
        daemon_url="http://127.0.0.1:8765",
        bearer_token="testtoken",
        scope_handle="project:test",
        repo_path=tmp_path,
        client_ids=[],
    )
    # Custom content should be preserved
    assert "Don't touch this." in claude_md.read_text()


def test_sync_merges_existing_cursor_config(tmp_path: Path, monkeypatch) -> None:
    from skein.sync import sync_all
    from skein import connections as conns
    monkeypatch.setattr(conns, "CONNECTIONS_PATH", tmp_path / "connections.json")

    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    existing = {"mcpServers": {"other-tool": {"url": "http://other"}}}
    (cursor_dir / "mcp.json").write_text(json.dumps(existing))

    sync_all(
        daemon_url="http://127.0.0.1:8765",
        bearer_token="testtoken",
        scope_handle="project:test",
        repo_path=tmp_path,
        client_ids=["cursor"],
    )

    data = json.loads((cursor_dir / "mcp.json").read_text())
    # Both entries should be present
    assert "other-tool" in data["mcpServers"]
    assert "skein" in data["mcpServers"]
