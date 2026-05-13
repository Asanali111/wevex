"""Tests for the passive docs scanner (iter 19)."""
from __future__ import annotations

from pathlib import Path

import pytest

from skein.docs_watcher import scan_docs
from skein.embeddings import HashEmbeddingProvider
from skein.models import IdentityCreate, ScopeCreate
from skein.passive import promote_scanned_facts
from skein.storage import Storage

# ---------------------------------------------------------------------------
# Basic discovery & shape
# ---------------------------------------------------------------------------


def test_scan_docs_empty_repo(tmp_path: Path) -> None:
    """An empty tmp dir returns no facts."""
    assert scan_docs(tmp_path) == []


def test_scan_docs_missing_dir(tmp_path: Path) -> None:
    """Non-existent directory returns no facts (no crash)."""
    assert scan_docs(tmp_path / "does-not-exist") == []


def test_scan_docs_readme_short(tmp_path: Path) -> None:
    """A short README emits one state fragment with stable topic_key."""
    (tmp_path / "README.md").write_text(
        "# My Project\n\nA short tagline that describes things.\n"
    )
    facts = scan_docs(tmp_path)
    assert len(facts) == 1
    f = facts[0]
    assert f.type == "state"
    assert "readme" in f.tags
    assert "docs" in f.tags
    assert f.topic_key == "docs:README"
    assert f.source_file == "README.md"
    assert "My Project" in f.content


# ---------------------------------------------------------------------------
# Heading split
# ---------------------------------------------------------------------------


def test_scan_docs_readme_long_splits_by_heading(tmp_path: Path) -> None:
    """A long README with multiple H1/H2 headings emits one fragment per
    heading, tagged with the kebab-slug of the heading text."""
    body_padding = "lorem ipsum " * 80  # ensures we cross SHORT_FILE_CHARS
    readme = (
        "# Overview\n\n"
        f"{body_padding}\n\n"
        "## Installation\n\n"
        f"{body_padding}\n\n"
        "## Quick Start Guide\n\n"
        f"{body_padding}\n"
    )
    (tmp_path / "README.md").write_text(readme)
    facts = scan_docs(tmp_path)
    assert len(facts) == 3

    topics = [f.topic_key for f in facts]
    assert "docs:README:overview" in topics
    assert "docs:README:installation" in topics
    assert "docs:README:quick-start-guide" in topics

    # Tag derived from heading slug should be present so `recall "installation"`
    # can hit the installation section.
    install_fact = next(f for f in facts if f.topic_key.endswith(":installation"))
    assert "installation" in install_fact.tags
    assert install_fact.type == "state"

    # All facts should be docs-tagged.
    for f in facts:
        assert "docs" in f.tags
        assert f.source_file == "README.md"


# ---------------------------------------------------------------------------
# ADR
# ---------------------------------------------------------------------------


def test_scan_docs_adr_typed_as_decision(tmp_path: Path) -> None:
    """A file under docs/adr/ named ``ADR-NNNN-…`` becomes a decision."""
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    (adr_dir / "ADR-0001-use-sqlite.md").write_text(
        "# ADR-0001: Use SQLite for the local store\n\n"
        "## Context\n\nWe need a single-binary embeddable DB.\n\n"
        "## Decision\n\nSQLite. Cheap, ubiquitous, ACID.\n"
    )
    facts = scan_docs(tmp_path)
    assert facts, "expected at least one fact from the ADR"
    for f in facts:
        assert f.type == "decision"
        assert "adr" in f.tags
        assert "ADR-0001" in f.tags
        # ADR confidence is higher than the default-not-allowed threshold.
        assert f.confidence >= 0.90


# ---------------------------------------------------------------------------
# CHANGELOG
# ---------------------------------------------------------------------------


def test_scan_docs_changelog_parses_entries(tmp_path: Path) -> None:
    """A CHANGELOG with two ``## `` entries becomes two changelog facts."""
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## 1.2.0 - 2026-05-12\n\n"
        "- Added docs_watcher.\n\n"
        "## 1.1.0 - 2026-04-30\n\n"
        "- Initial release.\n"
    )
    facts = scan_docs(tmp_path)
    assert len(facts) == 2
    topic_keys = [f.topic_key for f in facts]
    assert all(tk.startswith("changelog:") for tk in topic_keys), topic_keys
    # Tags include 'changelog' so recall can filter to release-note content.
    for f in facts:
        assert "changelog" in f.tags


# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------


def test_scan_docs_skips_huge_file(tmp_path: Path) -> None:
    """Files over MAX_FILE_BYTES (100 KB) are skipped — no fragment emitted."""
    (tmp_path / "README.md").write_text("x" * (200 * 1024))
    assert scan_docs(tmp_path) == []


# ---------------------------------------------------------------------------
# .gitignore
# ---------------------------------------------------------------------------


def test_scan_docs_respects_gitignore(tmp_path: Path) -> None:
    """An entry like ``docs/private/`` keeps that subtree out of the scan."""
    (tmp_path / ".gitignore").write_text("docs/private/\n")
    (tmp_path / "docs" / "private").mkdir(parents=True)
    (tmp_path / "docs" / "private" / "secret.md").write_text("# Secret\n\nDon't index me.\n")
    (tmp_path / "docs" / "public.md").write_text("# Public\n\nIndex me.\n")

    facts = scan_docs(tmp_path)
    sources = {f.source_file for f in facts}
    assert "docs/public.md" in sources
    assert not any("secret" in s for s in sources)


# ---------------------------------------------------------------------------
# Determinism / supersede
# ---------------------------------------------------------------------------


def test_scan_docs_topic_key_stable_across_runs(tmp_path: Path) -> None:
    """Re-running scan_docs yields the same topic_keys (supports supersede)."""
    (tmp_path / "README.md").write_text(
        "# Project\n\nFirst tagline.\n"
    )
    keys1 = sorted(f.topic_key for f in scan_docs(tmp_path) if f.topic_key)

    (tmp_path / "README.md").write_text(
        "# Project\n\nSecond tagline — content changed but topic_key stays.\n"
    )
    keys2 = sorted(f.topic_key for f in scan_docs(tmp_path) if f.topic_key)
    assert keys1 == keys2 == ["docs:README"]


# ---------------------------------------------------------------------------
# Integration with promote_scanned_facts
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_setup(tmp_path: Path):
    db_path = tmp_path / "p.db"
    storage = Storage(str(db_path))
    ident = storage.get_or_create_identity(
        IdentityCreate(handle="user:t", type="user", name="t")
    )
    scope = storage.create_scope(
        ScopeCreate(handle="project:p", type="project", name="p", owner_id=ident.id)
    )
    provider = HashEmbeddingProvider()
    yield storage, scope, ident, provider
    storage.close()


def test_docs_promote_supersedes_on_second_run(
    tmp_path: Path, storage_setup
) -> None:
    """Running scan_docs + promote_scanned_facts twice with different README
    contents superseded the first fragment instead of duplicating it."""
    storage, scope, ident, provider = storage_setup

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Skein\n\nOriginal tagline.\n")

    res1 = promote_scanned_facts(
        scan_docs(repo),
        storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id,
        source_tool="docs-scanner",
    )
    assert res1.auto_promoted == 1

    # Mutate the README content; topic_key must remain stable.
    (repo / "README.md").write_text("# Skein\n\nUpdated tagline with new content.\n")

    res2 = promote_scanned_facts(
        scan_docs(repo),
        storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id,
        source_tool="docs-scanner",
    )
    assert res2.auto_promoted == 1
    assert res2.superseded == 1

    live = storage.list_fragments(scope_id=scope.id, limit=10)
    assert len(live) == 1
    assert "Updated tagline" in live[0].content
    assert (live[0].metadata or {}).get("topic_key") == "docs:README"
