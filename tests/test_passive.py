"""Tests for the passive promotion pipeline (iter 14.1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from skein.embeddings import HashEmbeddingProvider
from skein.models import IdentityCreate, ScopeCreate
from skein.passive import promote_scanned_facts
from skein.scanner import ScannedFact
from skein.storage import Storage


@pytest.fixture
def storage_setup(tmp_path):
    """Fresh DB + identity + scope + provider."""
    db_path = tmp_path / "p.db"
    storage = Storage(str(db_path))
    ident = storage.get_or_create_identity(
        IdentityCreate(handle="user:t", type="user", name="t")
    )
    scope = storage.create_scope(
        ScopeCreate(handle="project:p", type="project",
                    name="p", owner_id=ident.id)
    )
    provider = HashEmbeddingProvider()
    yield storage, scope, ident, provider
    storage.close()


def test_promote_routes_by_confidence(storage_setup) -> None:
    storage, scope, ident, provider = storage_setup
    facts = [
        ScannedFact(content="High-conf fact", confidence=0.95),       # → auto
        ScannedFact(content="Medium-conf fact", confidence=0.70),     # → queue
        ScannedFact(content="Low-conf fact", confidence=0.30),        # → discard
    ]
    res = promote_scanned_facts(
        facts, storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res.auto_promoted == 1
    assert res.queued == 1
    assert res.discarded == 1
    # Verify the auto-promoted fact carries provenance
    frags = storage.list_fragments(scope_id=scope.id, limit=10)
    assert len(frags) == 1
    f = frags[0]
    assert f.created_by_tool == "code-scanner"
    assert f.extraction_method == "code-scanner"
    assert f.extraction_confidence == 0.95
    # Verify the queued candidate is pending
    cands = storage.list_extraction_candidates(scope_id=scope.id)
    assert len(cands) == 1
    assert cands[0]["status"] == "pending"
    assert cands[0]["confidence"] == 0.70


def test_promote_dedupes_identical_auto_facts(storage_setup) -> None:
    """Running the scanner twice shouldn't duplicate fragments."""
    storage, scope, ident, provider = storage_setup
    facts = [ScannedFact(content="dep: fastapi", confidence=0.95)]
    res1 = promote_scanned_facts(
        facts, storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    res2 = promote_scanned_facts(
        facts, storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res1.auto_promoted == 1
    assert res2.auto_promoted == 0
    assert res2.duplicate == 1
    assert len(storage.list_fragments(scope_id=scope.id, limit=10)) == 1


def test_promote_dedupes_queue_candidates(storage_setup) -> None:
    storage, scope, ident, provider = storage_setup
    facts = [ScannedFact(content="mid conf", confidence=0.70)]
    promote_scanned_facts(
        facts, storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    res2 = promote_scanned_facts(
        facts, storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res2.queued == 0
    assert res2.duplicate == 1
    assert len(storage.list_extraction_candidates(scope_id=scope.id)) == 1


def test_promote_empty_returns_zeroed_result(storage_setup) -> None:
    storage, scope, ident, provider = storage_setup
    res = promote_scanned_facts(
        [], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res.auto_promoted == res.queued == res.discarded == res.duplicate == 0
