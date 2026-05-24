"""Tests for the iter-27 embedding-provider defaults and the bm25 provider.

iter 27 removed the Gemini embedding provider — see
``tests/test_embeddings_naming.py`` for the alias / naming-guard tests.
"""
from __future__ import annotations

import pytest

from skein.embeddings import (
    BM25OnlyProvider,
    EmbeddingProvider,
    FastembedProvider,
    HashEmbeddingProvider,
    best_available_provider_name,
    get_provider,
)


def test_bm25_provider_returns_zero_vectors() -> None:
    p = BM25OnlyProvider()
    # iter 27: bm25 dimension matches the fastembed default (384) so a
    # downgrade path doesn't silently invalidate stored embeddings via
    # bytes_to_vec's dimension-mismatch zero fallback.
    assert p.dimension == 384
    assert p.is_real is False
    vec = p.embed_one("anything")
    assert vec == [0.0] * 384


def test_bm25_provider_handles_batches() -> None:
    p = BM25OnlyProvider()
    vecs = p.embed(["one", "two", "three"])
    assert len(vecs) == 3
    assert all(v == [0.0] * 384 for v in vecs)


def test_hash_is_not_real() -> None:
    """The legacy hash provider must report is_real=False so doctor warns."""
    assert HashEmbeddingProvider().is_real is False


def test_fastembed_is_real() -> None:
    """Class attribute set on the class — avoids loading the ONNX runtime
    in tests that only care about capability advertising."""
    assert FastembedProvider.is_real is True


def test_get_provider_supports_bm25_aliases() -> None:
    for name in ("bm25", "BM25", "none", "off"):
        p = get_provider(name)
        assert isinstance(p, BM25OnlyProvider)


def test_get_provider_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_provider("magic")


def test_best_available_is_unconditionally_fastembed(monkeypatch) -> None:
    """iter 27: the picker no longer auto-detects cloud keys. fastembed is
    the unconditional default for new installs — local, zero-config, no
    API key. Users who want cloud SOTA opt into 'openai' explicitly."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert best_available_provider_name() == "fastembed"


def test_best_available_ignores_cloud_keys(monkeypatch) -> None:
    """Even with OPENAI_API_KEY in the env, default stays fastembed. Cloud
    is opt-in via explicit config, not env-key auto-detection — the old
    auto-detect produced silent failures when keys were stale / depleted
    (cf. iter 27 daemon-wedge incident)."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    assert best_available_provider_name() == "fastembed"


def test_provider_base_is_not_real() -> None:
    """The abstract base must declare is_real=False so subclasses default safe."""
    assert EmbeddingProvider.is_real is False
