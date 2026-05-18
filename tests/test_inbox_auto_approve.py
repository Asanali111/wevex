"""Tests for ``skein inbox auto-approve`` (iter 24).

The CLI lets the user bleed the extraction-candidate queue back down without
manually clicking through every item — the 173-item backlog the project had
accumulated was poisoning recall scores because the high-confidence facts
never made it into the search index. These tests pin the safety semantics:
* ``--min-confidence`` filters before promoting
* ``--dry-run`` writes nothing
* rejected/already-approved candidates are skipped
* successful runs flip the candidate to "approved" and surface fragments to
  the recall path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skein.cli import main
from skein.storage import Storage


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch):
    """Point the CLI at a writable temp config + DB.

    The CLI loads `SkeinConfig` from disk on every invocation, so we redirect
    ``$SKEIN_CONFIG`` to a temp file and pin the embedding provider to the
    deterministic hash backend.
    """
    db_path = tmp_path / "skein.db"
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "db_path": str(db_path),
        "bearer_token": "x" * 64,
        "embedding_provider": "hash",
        "default_scope": "project:test",
        "embedding_dimension": 64,
    }))
    monkeypatch.setenv("SKEIN_CONFIG", str(cfg_path))
    monkeypatch.setenv("SKEIN_EMBEDDING_PROVIDER", "hash")

    # Seed a scope + a few extraction candidates spanning confidences
    from skein.config import reset_config
    reset_config(None)  # force re-read on next CLI call

    storage = Storage(str(db_path))
    try:
        from skein.models import IdentityCreate, ScopeCreate
        user = storage.create_identity(IdentityCreate(
            handle="user:test", type="user", name="Test",
        ))
        scope = storage.create_scope(ScopeCreate(
            handle="project:test", type="project", name="Test",
            owner_id=user.id,
        ))
        for i, conf in enumerate([0.95, 0.90, 0.70, 0.55]):
            storage.add_extraction_candidate(
                scope_id=scope.id,
                content=f"candidate-{i} content body",
                type="fact",
                confidence=conf,
                source_tool="test-scanner",
            )
    finally:
        storage.close()
    return db_path


def _list_pending(db_path: Path) -> int:
    st = Storage(str(db_path))
    try:
        return st.count_extraction_candidates(status="pending")
    finally:
        st.close()


def test_auto_approve_promotes_high_confidence_only(cli_db: Path) -> None:
    assert _list_pending(cli_db) == 4
    runner = CliRunner()
    result = runner.invoke(main, [
        "inbox", "auto-approve", "--min-confidence", "0.85",
    ])
    assert result.exit_code == 0, result.output
    # 0.95 and 0.90 promoted; 0.70 and 0.55 left behind.
    assert _list_pending(cli_db) == 2
    assert "Promoted 2" in result.output


def test_auto_approve_dry_run_writes_nothing(cli_db: Path) -> None:
    assert _list_pending(cli_db) == 4
    runner = CliRunner()
    result = runner.invoke(main, [
        "inbox", "auto-approve", "--min-confidence", "0.50", "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    assert _list_pending(cli_db) == 4
    assert "Dry run" in result.output


def test_auto_approve_threshold_excludes_everything(cli_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "inbox", "auto-approve", "--min-confidence", "0.99",
    ])
    assert result.exit_code == 0, result.output
    assert _list_pending(cli_db) == 4
    assert "No pending candidates" in result.output


def test_auto_approve_min_age_days_excludes_young(cli_db: Path) -> None:
    """Age filter must actually fire — silently parsing SQLite's space-separated
    timestamp wrong on Python 3.9/3.10 would make this a no-op."""
    # Every candidate seeded in ``cli_db`` was created just now, so any
    # positive --min-age-days should exclude them all.
    runner = CliRunner()
    result = runner.invoke(main, [
        "inbox", "auto-approve",
        "--min-confidence", "0.85", "--min-age-days", "1",
    ])
    assert result.exit_code == 0, result.output
    assert _list_pending(cli_db) == 4
    assert "No pending candidates" in result.output


def test_auto_approve_min_age_days_includes_old(cli_db: Path) -> None:
    """Backdate one candidate and confirm --min-age-days=1 lets it through."""
    st = Storage(str(cli_db))
    try:
        # Rewrite created_at on the 0.95-confidence candidate to two days ago,
        # using the same space-separated format SQLite's datetime('now') emits.
        st._conn.execute(
            """UPDATE extraction_candidates
               SET created_at = datetime('now', '-2 days')
               WHERE confidence = 0.95"""
        )
        st._conn.commit()
    finally:
        st.close()

    runner = CliRunner()
    result = runner.invoke(main, [
        "inbox", "auto-approve",
        "--min-confidence", "0.85", "--min-age-days", "1",
    ])
    assert result.exit_code == 0, result.output
    # Only the backdated 0.95 candidate should promote; the day-zero 0.90
    # candidate is still too young.
    assert _list_pending(cli_db) == 3
    assert "Promoted 1" in result.output


def test_auto_approve_creates_real_fragments(cli_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "inbox", "auto-approve", "--min-confidence", "0.85",
    ])
    assert result.exit_code == 0, result.output
    # Verify the promoted candidates are now live fragments.
    st = Storage(str(cli_db))
    try:
        rows = st._conn.execute(
            "SELECT COUNT(*) FROM fragments WHERE created_by_tool = ?",
            ("test-scanner",),
        ).fetchone()
        assert rows[0] == 2
    finally:
        st.close()
