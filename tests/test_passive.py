"""Tests for the passive promotion pipeline (iter 14.1)."""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# iter 18: topic_key-based supersede (prevents "Tests live in tests/ (38)"
# and "Tests live in tests/ (29)" from coexisting in AGENTS.md)
# ---------------------------------------------------------------------------


def test_rescan_with_changed_count_supersedes_old(storage_setup) -> None:
    """Two scans of the same topic with different content → only one surfaces."""
    storage, scope, ident, provider = storage_setup
    first = ScannedFact(
        content="Tests live in `tests/` (29 files).",
        type="preference", confidence=0.95,
        tags=["testing", "layout"], topic_key="tests-layout",
    )
    second = ScannedFact(
        content="Tests live in `tests/` (38 files).",
        type="preference", confidence=0.95,
        tags=["testing", "layout"], topic_key="tests-layout",
    )

    res1 = promote_scanned_facts(
        [first], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res1.auto_promoted == 1

    res2 = promote_scanned_facts(
        [second], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res2.auto_promoted == 1
    assert res2.superseded == 1

    # AGENTS.md renders live preferences via list_fragments(include_stale=False)
    live = storage.list_fragments(
        scope_id=scope.id, type_filter="preference", limit=10,
    )
    assert len(live) == 1
    assert "38 files" in live[0].content

    # The old fragment is stale, and supersede chain points both directions
    all_frags = storage.list_fragments(
        scope_id=scope.id, type_filter="preference", limit=10,
        include_stale=True,
    )
    assert len(all_frags) == 2
    old = next(f for f in all_frags if "29 files" in f.content)
    new = next(f for f in all_frags if "38 files" in f.content)
    assert old.is_stale is True
    assert old.stale_reason and new.id in old.stale_reason
    assert old.superseded_by_fragment_id == new.id
    assert new.supersedes_fragment_id == old.id
    assert (new.metadata or {}).get("topic_key") == "tests-layout"


def test_rescan_legacy_unkeyed_duplicates_get_consolidated(storage_setup) -> None:
    """Pre-iter-18 DB state: two un-keyed duplicate fragments for one fact slot.

    On the first scan after upgrade, the new fact carries a topic_key but the
    legacy fragments have none. The content-stem fingerprint must still match
    them so both get superseded — otherwise AGENTS.md keeps showing the bug
    forever.

    Seed the duplicates by writing directly to storage, the way the pre-fix
    scanner left them.
    """
    storage, scope, ident, provider = storage_setup
    from skein.models import CommitCreate as CC
    from skein.models import FragmentCreate as FC

    commit = storage.create_commit(CC(
        author_id=ident.id, scope_id=scope.id, message="seed legacy",
    ))
    for content in (
        "Tests live in `tests/` (29 files).",
        "Tests live in `tests/` (38 files).",
    ):
        storage.create_fragment(FC(
            content=content, type="preference",
            scope_id=scope.id, owner_id=ident.id,
            tags=["testing", "layout"],
            created_by_tool="code-scanner",
            extraction_method="code-scanner", extraction_confidence=0.95,
        ), commit_id=commit.id)

    pre = storage.list_fragments(
        scope_id=scope.id, type_filter="preference", limit=10,
    )
    assert len(pre) == 2
    assert all(not (f.metadata or {}).get("topic_key") for f in pre)

    # Now: post-upgrade scan with topic_key set
    fixed = ScannedFact(
        content="Tests live in `tests/` (31 files).",
        type="preference", confidence=0.95,
        tags=["testing", "layout"], topic_key="tests-layout",
    )
    res = promote_scanned_facts(
        [fixed], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res.auto_promoted == 1
    assert res.superseded == 2  # both legacy fragments retired

    live = storage.list_fragments(
        scope_id=scope.id, type_filter="preference", limit=10,
    )
    assert len(live) == 1
    assert "31 files" in live[0].content
    assert (live[0].metadata or {}).get("topic_key") == "tests-layout"


def test_unchanged_topic_doesnt_create_new_fragment(storage_setup) -> None:
    """Same topic_key + same content on re-scan = no-op (true duplicate)."""
    storage, scope, ident, provider = storage_setup
    fact = ScannedFact(
        content="Linter: ruff (configured in pyproject.toml).",
        type="preference", confidence=0.95,
        tags=["linting", "ruff"], topic_key="python-linter",
    )
    promote_scanned_facts(
        [fact], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    res2 = promote_scanned_facts(
        [fact], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res2.auto_promoted == 0
    assert res2.duplicate == 1
    assert res2.superseded == 0
    assert len(storage.list_fragments(scope_id=scope.id, limit=10)) == 1


def test_distinct_topics_with_same_stem_dont_cross_supersede(storage_setup) -> None:
    """Two facts whose contents share a stem (e.g. two EXPOSE ports) must NOT
    supersede each other. Without this guard, every re-scan would churn:
    f1's stem match drags in f2 (and vice versa), marking both stale and
    creating two new IDs per scan.
    """
    storage, scope, ident, provider = storage_setup
    f1 = ScannedFact(
        content="Service exposes port `8000` (Dockerfile EXPOSE).",
        confidence=0.95, tags=["docker", "network"],
        topic_key="docker-expose:8000",
    )
    f2 = ScannedFact(
        content="Service exposes port `9000` (Dockerfile EXPOSE).",
        confidence=0.95, tags=["docker", "network"],
        topic_key="docker-expose:9000",
    )
    promote_scanned_facts(
        [f1, f2], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    ids_before = {f.id for f in storage.list_fragments(scope_id=scope.id, limit=20)}

    res = promote_scanned_facts(
        [f1, f2], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    ids_after = {f.id for f in storage.list_fragments(scope_id=scope.id, limit=20)}
    assert res.auto_promoted == 0
    assert res.superseded == 0
    assert ids_before == ids_after  # no churn


def test_unchanged_content_retires_legacy_duplicate(storage_setup) -> None:
    """When the DB already has two fragments with identical content
    (pre-fix duplicate bug), the next scan with that same content keeps
    one and marks the other stale rather than stacking a third copy.
    """
    storage, scope, ident, provider = storage_setup
    # Seed two identical legacy fragments by stubbing the dedup table — the
    # safest way is to write directly, bypassing promote.
    from skein.models import CommitCreate as CC
    from skein.models import FragmentCreate as FC
    commit = storage.create_commit(CC(
        author_id=ident.id, scope_id=scope.id, message="seed",
    ))
    common = dict(
        content="Linter: ruff (configured in pyproject.toml).",
        type="preference", scope_id=scope.id, owner_id=ident.id,
        tags=["linting", "ruff"], created_by_tool="code-scanner",
        extraction_method="code-scanner", extraction_confidence=0.95,
    )
    storage.create_fragment(FC(**common), commit_id=commit.id)
    storage.create_fragment(FC(**common), commit_id=commit.id)

    fact = ScannedFact(
        content="Linter: ruff (configured in pyproject.toml).",
        type="preference", confidence=0.95,
        tags=["linting", "ruff"], topic_key="python-linter",
    )
    res = promote_scanned_facts(
        [fact], storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id, source_tool="code-scanner",
    )
    assert res.auto_promoted == 0
    assert res.duplicate == 1
    live = storage.list_fragments(scope_id=scope.id, limit=10)
    assert len(live) == 1
