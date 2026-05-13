"""End-to-end CLI tests that the .skein/scope pin is honored by every command.

This is a regression test for the bug where ``skein ingest .`` ignored the
project-level scope pin and silently used ``cfg.default_scope``, causing
chunks to land in the wrong scope.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from skein.cli import main as skein_cli
from skein.config import SkeinConfig, reset_config
from skein.dependencies import set_provider, set_storage
from skein.storage import Storage


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Set up a fresh DB, config, and a project dir with a scope pin."""
    db_path = tmp_path / "skein.db"
    cfg_path = tmp_path / "config.json"

    cfg = SkeinConfig({
        "db_path": str(db_path),
        "bearer_token": "x" * 64,
        "embedding_provider": "hash",
        "default_scope": "project:default",
    })
    cfg.save(cfg_path)
    monkeypatch.setenv("SKEIN_CONFIG", str(cfg_path))
    # Suppress the "using scope X from .skein/scope" hint so JSON output stays clean
    monkeypatch.setenv("SKEIN_QUIET_PIN", "1")
    reset_config(None)  # force re-read from $SKEIN_CONFIG
    monkeypatch.delenv("SKEIN_SCOPE", raising=False)

    # Clear dependency singletons so CLI freshly opens its own
    set_storage(None)  # type: ignore[arg-type]
    set_provider(None)  # type: ignore[arg-type]

    # Project dir with a .skein/scope pin
    proj = tmp_path / "myproj"
    (proj / ".skein").mkdir(parents=True)
    (proj / ".skein" / "scope").write_text("project:pinned-here\n")

    # Add a sample file to ingest
    (proj / "src").mkdir()
    (proj / "src" / "auth.py").write_text(
        "def login(user, pw):\n    return issue_bearer_token(user)\n"
    )

    monkeypatch.chdir(proj)
    yield {"tmp_path": tmp_path, "db_path": db_path, "proj": proj, "cfg": cfg}
    reset_config(None)


def test_ingest_honors_scope_pin(cli_env):
    """Regression: `skein ingest .` (no --scope) must use .skein/scope, not config default."""
    runner = CliRunner()
    result = runner.invoke(skein_cli, ["ingest", ".", "--quiet"])
    assert result.exit_code == 0, f"exit {result.exit_code}\nstdout:\n{result.output}"

    # The pinned scope should now exist with chunks; the config default should NOT
    s = Storage(str(cli_env["db_path"]))
    pinned = s.get_scope("project:pinned-here")
    default = s.get_scope("project:default")
    assert pinned is not None, "pinned scope was not created"
    pinned_chunks = s.list_chunks(scope_id=pinned.id, limit=100)
    assert len(pinned_chunks) >= 1, "expected at least one chunk in pinned scope"
    if default is not None:
        default_chunks = s.list_chunks(scope_id=default.id, limit=100)
        assert default_chunks == [], (
            "chunks leaked into default scope — pin not honored"
        )
    s.close()


def test_ingest_explicit_scope_overrides_pin(cli_env):
    """When --scope is passed, it wins over the pin."""
    runner = CliRunner()
    result = runner.invoke(
        skein_cli,
        ["ingest", ".", "--quiet", "--scope", "project:explicit"],
    )
    assert result.exit_code == 0, result.output

    s = Storage(str(cli_env["db_path"]))
    explicit = s.get_scope("project:explicit")
    pinned = s.get_scope("project:pinned-here")
    assert explicit is not None
    assert s.list_chunks(scope_id=explicit.id, limit=100), \
        "expected chunks in explicit scope"
    if pinned is not None:
        assert s.list_chunks(scope_id=pinned.id, limit=100) == [], \
            "pinned scope got chunks despite explicit --scope override"
    s.close()


def test_chunks_stats_honors_pin(cli_env):
    """`skein chunks stats` should also honor the pin."""
    runner = CliRunner()
    runner.invoke(skein_cli, ["ingest", ".", "--quiet"])

    result = runner.invoke(skein_cli, ["chunks", "stats", "--json"])
    assert result.exit_code == 0, result.output
    # The output should reflect the pinned scope's stats
    data = json.loads(result.output)
    assert data["total_chunks"] >= 1


def test_search_honors_pin(cli_env):
    """`skein search` should look in the pinned scope."""
    runner = CliRunner()
    runner.invoke(skein_cli, ["ingest", ".", "--quiet"])
    result = runner.invoke(
        skein_cli, ["search", "bearer token", "--limit", "3", "--no-show-content"],
    )
    assert result.exit_code == 0, result.output
    # Either we get hits from the pinned scope, or "no chunks matched"
    # but never a complaint about a different scope being missing
    assert "Scope" not in result.output or "not found" not in result.output
