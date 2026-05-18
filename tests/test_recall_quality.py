"""Tests for the iter-24 recall quality signal.

The previous public score (RRF fused) caps at ~0.033 with two ranked lists at
k=60, so the historical AGENTS.md threshold ``score < 0.1`` could never trigger.
This module pins the replacement signal:

* ``RecallResult.cosine`` mirrors the underlying vector similarity when a
  vector hit contributed.
* ``RecallResult.bm25`` mirrors the underlying BM25 relevance when a keyword
  hit contributed.
* ``RecallResult.quality`` buckets the result so a caller can route on a
  human-readable label without knowing what an RRF score looks like.
"""
from __future__ import annotations

import pytest

from skein.embeddings import HashEmbeddingProvider, vec_to_bytes
from skein.models import (
    FragmentCreate, RecallRequest, classify_recall_quality,
)
from skein.retrieval import _rrf_fuse, recall
from skein.storage import Storage


# ---------------------------------------------------------------------------
# classify_recall_quality — pure unit
# ---------------------------------------------------------------------------

def test_quality_high_when_cosine_strong() -> None:
    assert classify_recall_quality(
        cosine=0.85, matched_by="hybrid", rank=1,
    ) == "high"


def test_quality_medium_band() -> None:
    assert classify_recall_quality(
        cosine=0.55, matched_by="vector", rank=2,
    ) == "medium"


def test_quality_low_band() -> None:
    assert classify_recall_quality(
        cosine=0.40, matched_by="vector", rank=3,
    ) == "low"


def test_quality_none_when_cosine_floor() -> None:
    assert classify_recall_quality(
        cosine=0.10, matched_by="vector", rank=1,
    ) == "none"


def test_quality_keyword_only_top_three_is_low() -> None:
    """No vector signal but the keyword search ranked it high — give it a
    "low" floor so the caller sees it without being misled into trusting it."""
    assert classify_recall_quality(
        cosine=None, matched_by="keyword", rank=2,
    ) == "low"


def test_quality_keyword_only_below_top_is_none() -> None:
    assert classify_recall_quality(
        cosine=None, matched_by="keyword", rank=8,
    ) == "none"


# ---------------------------------------------------------------------------
# _rrf_fuse — raw scores must survive fusion
# ---------------------------------------------------------------------------

def test_rrf_fuse_preserves_raw_scores() -> None:
    """The fused tuples must carry a ``{list_name: raw_score}`` map so the
    downstream caller can attach the underlying signals to RecallResult."""
    fused = _rrf_fuse(
        lists=[
            [("a", 1.5), ("b", 0.8)],            # "keyword" raw is BM25
            [("a", 0.72), ("c", 0.41)],          # "vector" raw is cosine
        ],
        list_names=["keyword", "vector"],
    )
    by_id = {item[0]: item for item in fused}

    a = by_id["a"]
    assert a[2] == "hybrid"
    assert a[3]["keyword"] == pytest.approx(1.5)
    assert a[3]["vector"] == pytest.approx(0.72)

    b = by_id["b"]
    assert b[2] == "keyword"
    assert b[3] == {"keyword": pytest.approx(0.8)}

    c = by_id["c"]
    assert c[2] == "vector"
    assert c[3] == {"vector": pytest.approx(0.41)}


# ---------------------------------------------------------------------------
# End-to-end — recall populates the new fields
# ---------------------------------------------------------------------------

@pytest.fixture
def filled_storage(seeded_storage: Storage) -> Storage:
    """Reused minimal seed so the recall pass has something to rank."""
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user
    provider = HashEmbeddingProvider()

    rows = [
        ("decision", "use Redis for caching", "backend/cache"),
        ("decision", "use PostgreSQL for primary storage", "backend/db"),
        ("fact",     "Redis default TTL is 0 (no expiry)", "backend/cache"),
        ("state",    "the auth middleware uses bcrypt", "backend/auth"),
    ]
    for frag_type, content, territory in rows:
        vec = provider.embed_one(content)
        st.create_fragment(
            FragmentCreate(
                type=frag_type, content=content,
                scope_id=scope.id, owner_id=user.id, territory=territory,
            ),
            embedding=vec_to_bytes(vec),
        )
    return st


def test_recall_populates_quality_and_raw_signals(filled_storage: Storage) -> None:
    provider = HashEmbeddingProvider()
    scope = filled_storage._test_scope

    req = RecallRequest(query="Redis caching", scope=scope.handle, limit=5)
    resp = recall(req, filled_storage, provider)

    assert resp.total > 0
    for r in resp.results:
        assert r.quality in ("high", "medium", "low", "none")
        # At least one of the raw signals must be populated unless the
        # match was a pure keyword-fallback with no embedding.
        if r.matched_by in ("hybrid", "vector"):
            assert r.cosine is not None
        if r.matched_by in ("hybrid", "keyword"):
            assert r.bm25 is not None
