"""Tests for retrieval.py — hybrid BM25 + vector + RRF."""
from __future__ import annotations

import pytest

from skein.embeddings import HashEmbeddingProvider, vec_to_bytes
from skein.models import (
    FragmentCreate,
    RecallRequest,
)
from skein.retrieval import _rrf_fuse, recall
from skein.storage import Storage


@pytest.fixture
def filled_storage(seeded_storage: Storage) -> Storage:
    """Storage with several fragments across different types."""
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user
    provider = HashEmbeddingProvider()

    contents = [
        ("decision", "use Redis for caching", "backend/cache"),
        ("decision", "use PostgreSQL for primary storage", "backend/db"),
        ("fact",     "Redis default TTL is 0 (no expiry)", "backend/cache"),
        ("state",    "Redis is running on port 6379", "backend/cache"),
        ("preference", "prefer async Python over sync", "backend"),
        ("requirement", "all API responses must be under 200ms", "backend/api"),
        ("observation", "the auth middleware has a memory leak", "backend/auth"),
    ]

    for frag_type, content, territory in contents:
        vec = provider.embed_one(content)
        st.create_fragment(
            FragmentCreate(
                type=frag_type, content=content,
                scope_id=scope.id, owner_id=user.id,
                territory=territory,
            ),
            embedding=vec_to_bytes(vec),
        )

    return st


# ---------------------------------------------------------------------------
# RRF unit tests
# ---------------------------------------------------------------------------

def test_rrf_fuse_empty() -> None:
    result = _rrf_fuse([[], []], list_names=["kw", "vec"])
    assert result == []


def test_rrf_fuse_single_list() -> None:
    result = _rrf_fuse(
        lists=[[("a", 1.0), ("b", 0.9), ("c", 0.8)]],
        list_names=["kw"],
    )
    ids = [r[0] for r in result]
    assert ids == ["a", "b", "c"]
    assert all(r[2] == "kw" for r in result)


def test_rrf_fuse_two_lists() -> None:
    list_a = [("a", 1.0), ("b", 0.8), ("c", 0.6)]
    list_b = [("b", 0.9), ("a", 0.7), ("d", 0.5)]
    result = _rrf_fuse([list_a, list_b], list_names=["kw", "vec"])

    ids = [r[0] for r in result]
    # "a" and "b" appear in both lists — should dominate
    assert ids[0] in ("a", "b")
    assert ids[1] in ("a", "b")
    # Items appearing in both lists get "hybrid" source
    hybrid = {r[0] for r in result if r[2] == "hybrid"}
    assert "a" in hybrid
    assert "b" in hybrid


def test_rrf_score_descending() -> None:
    list_a = [("x", 1.0), ("y", 0.5)]
    list_b = [("y", 1.0), ("x", 0.5)]
    result = _rrf_fuse([list_a, list_b], list_names=["kw", "vec"])
    scores = [r[1] for r in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# End-to-end recall tests
# ---------------------------------------------------------------------------

def test_recall_returns_results(filled_storage: Storage) -> None:
    provider = HashEmbeddingProvider()
    scope = filled_storage._test_scope

    req = RecallRequest(query="caching Redis", scope=scope.handle, limit=5)
    resp = recall(req, filled_storage, provider)

    assert resp.total > 0
    assert len(resp.results) <= 5
    assert resp.query == "caching Redis"
    assert resp.scope == scope.handle


def test_recall_type_filter(filled_storage: Storage) -> None:
    provider = HashEmbeddingProvider()
    scope = filled_storage._test_scope

    req = RecallRequest(
        query="Redis caching",
        scope=scope.handle,
        types=["decision"],
        limit=10,
    )
    resp = recall(req, filled_storage, provider)
    for r in resp.results:
        assert r.fragment.type == "decision"


def test_recall_territory_filter(filled_storage: Storage) -> None:
    provider = HashEmbeddingProvider()
    scope = filled_storage._test_scope

    req = RecallRequest(
        query="storage database",
        scope=scope.handle,
        territory="backend/cache",
        limit=10,
    )
    resp = recall(req, filled_storage, provider)
    for r in resp.results:
        assert r.fragment.territory is not None
        assert r.fragment.territory.startswith("backend/cache")


def test_recall_scope_not_found(storage: Storage) -> None:
    provider = HashEmbeddingProvider()
    req = RecallRequest(
        query="anything",
        scope="project:nonexistent",
        limit=5,
    )
    resp = recall(req, storage, provider)
    assert resp.total == 0
    assert resp.results == []


def test_recall_empty_db(storage: Storage) -> None:
    provider = HashEmbeddingProvider()
    # Seed minimal identity + scope
    from skein.models import IdentityCreate, ScopeCreate
    user = storage.create_identity(IdentityCreate(
        handle="user:empty", type="user", name="Empty",
    ))
    scope = storage.create_scope(ScopeCreate(
        handle="project:empty", type="project", name="Empty", owner_id=user.id,
    ))
    req = RecallRequest(query="anything", scope=scope.handle, limit=5)
    resp = recall(req, storage, provider)
    assert resp.total == 0


def test_recall_result_ranking(filled_storage: Storage) -> None:
    """Results should have ascending rank values."""
    provider = HashEmbeddingProvider()
    scope = filled_storage._test_scope

    req = RecallRequest(query="backend storage", scope=scope.handle, limit=5)
    resp = recall(req, filled_storage, provider)

    if resp.results:
        ranks = [r.rank for r in resp.results]
        assert ranks == list(range(1, len(ranks) + 1))


def test_recall_scope_lineage(filled_storage: Storage) -> None:
    """A query on a child scope should return parent-scope fragments too."""
    from skein.models import ScopeCreate

    st = filled_storage
    parent_scope = st._test_scope
    parent_user = st._test_user

    child_scope = st.create_scope(ScopeCreate(
        handle="team:backend-child",
        type="team",
        name="Backend Child",
        owner_id=parent_user.id,
        parent_scope_id=parent_scope.id,
    ))

    # Fragment is in parent scope; query is on child
    provider = HashEmbeddingProvider()
    req = RecallRequest(query="Redis caching", scope=child_scope.handle, limit=10)
    resp = recall(req, st, provider)
    # Should find the Redis fragments from the parent scope
    assert resp.total > 0
