"""Embedding provider abstraction.

Supported providers:
  "fastembed" – local ONNX (BAAI/bge-small-en-v1.5, 384-dim). Default for new
                installs. No API key. ~130 MB one-time model download cached
                under ``~/.cache/fastembed/``.
  "openai"    – OpenAI text-embedding-3-small (1536-dim). Requires
                OPENAI_API_KEY env var and openai package.
  "bm25"      – Zero-vector no-op. Recall falls back to FTS5/BM25 only.
                Honest about being keyword-only.
  "hash"      – Deterministic SHA-256-derived pseudo-vector. Tests only —
                doctor warns when active in production.

All providers expose:
    embed(texts: list[str]) -> list[list[float]]
    embed_one(text: str)    -> list[float]
    dimension: int
    is_real:   bool   # True for semantic providers (fastembed, openai)

----------------------------------------------------------------------------
NAMING — read before adding any "gemini" code here.
----------------------------------------------------------------------------
There is no Gemini *embedding* provider in Skein. The previous
``GeminiEmbeddingProvider`` was removed in iter 27 because:
  - the free tier rate-limits hard and parks the asyncio event loop in retry
    loops, wedging ``/health`` and forcing launchd respawn cascades;
  - fastembed (BGE-small) is local, has no API key story, and is within ~4
    MTEB points of Gemini for our hybrid-RRF use case.

``"gemini"`` is still accepted as a config string and silently aliased to
``"fastembed"`` so existing user configs don't crash the daemon on upgrade.
The alias logs a one-time deprecation warning.

The string ``"gemini"`` in this codebase always refers to the embedding API.
It is NOT the same as the **Gemini CLI** LLM client at
``skein/clients.py::GeminiCLIClient`` (id ``"gemini_cli"``). That client is
a sync target — Skein writes an MCP config snippet so the ``gemini`` CLI
binary can connect to the local daemon, the same way it does for Claude
Code, Cursor, Codex, etc. The Gemini CLI client is fully supported and
unrelated to embeddings. ``tests/test_embeddings_naming.py`` enforces
both invariants.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import TYPE_CHECKING, Optional

# Numpy is lazy-loaded inside the functions that actually use it. At ~250 ms
# import cost, it dominates daemon boot when only a fraction of requests touch
# embeddings hot-paths. ``from __future__ import annotations`` (above) makes
# the `np.ndarray` type hints in this file no-ops at runtime, so we don't need
# the symbol at module top-level. Hot helpers (`vec_to_bytes`, `bytes_to_vec`,
# `cosine_similarity`) import numpy on first call — Python caches sys.modules
# so subsequent calls pay only a dict lookup.
if TYPE_CHECKING:
    import numpy as np  # noqa: F401  — purely for type-checker resolution

logger = logging.getLogger("skein.embeddings")


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Abstract base class."""

    dimension: int = 384
    # Subclasses set this. ``is_real = True`` means embeddings produced by
    # this provider are semantically meaningful (e.g. fastembed, OpenAI).
    # ``is_real = False`` means embeddings are pseudo-random or zero —
    # vector ranking is decorative for this provider, retrieval falls back
    # to FTS5/BM25. ``skein doctor`` and the README rely on this marker.
    is_real: bool = False
    # Stable identifier — matches the config string.
    name: str = "base"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# BM25-only provider (no vector — honest about being keyword-only)
# ---------------------------------------------------------------------------

class BM25OnlyProvider(EmbeddingProvider):
    """Returns zero vectors so the vector retrieval path is a true no-op.

    Search falls back to FTS5 (BM25 keyword matching) exclusively — fast,
    honest, doesn't pretend to be semantic. ``fastembed`` is the
    recommended default for semantic search; use this only when the local
    model can't be installed.
    """

    dimension: int = 384
    is_real: bool = False
    name: str = "bm25"

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Zero vectors. Stored embeddings will all be the zero vector,
        # which produces zero cosine similarity to everything — vector
        # search effectively becomes a no-op for ranking purposes.
        return [[0.0] * self.dimension for _ in texts]


# ---------------------------------------------------------------------------
# Hash provider (legacy: deterministic but non-semantic)
# ---------------------------------------------------------------------------

class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-embedding from SHA-256.

    Not semantically meaningful — exists so tests are reproducible without
    network access. **Not for production use.** ``skein doctor`` warns when
    this provider is active in a live config.
    """

    is_real: bool = False
    name: str = "hash"
    dimension: int = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = self._hash_to_vec(text)
            results.append(vec)
        return results

    @staticmethod
    def _hash_to_vec(text: str, dim: int = 768) -> list[float]:
        # Build a deterministic float vector from repeated SHA-256 hashing.
        seed = text.encode("utf-8")
        floats: list[float] = []
        counter = 0
        while len(floats) < dim:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            # Each byte gives one float in [-1, 1]
            for b in digest:
                floats.append((b / 127.5) - 1.0)
                if len(floats) == dim:
                    break
            counter += 1
        # Normalize to unit vector
        import numpy as np
        arr = np.array(floats, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()


# ---------------------------------------------------------------------------
# Fastembed provider (local, default — iter 23)
# ---------------------------------------------------------------------------

class FastembedProvider(EmbeddingProvider):
    """Local ONNX-quantized semantic embeddings via the `fastembed` package.

    Default for new installs (iter 23). Replaces the previous BM25-only
    default so `pip install skein && skein up` produces semantic search
    out-of-the-box with no API key and no cloud round-trip.

    Model: ``BAAI/bge-small-en-v1.5`` — 384-dim, ~62 MTEB retrieval avg
    (~4 points below Gemini ``gemini-embedding-001``, well above the
    older ``all-MiniLM-L6-v2``). The ONNX weights (~130 MB) download to
    ``~/.cache/fastembed/`` on first instantiation; subsequent runs are
    instant. Per-embed latency on M-series Macs is ~30–50 ms.

    Privacy: every embed call runs locally. No network egress, no API
    key, no per-request cost. Subject to neither rate limits nor cloud
    downtime — exactly the "local-first context bus" the README promises.
    """

    dimension: int = 384
    model: str = "BAAI/bge-small-en-v1.5"
    is_real: bool = True
    name: str = "fastembed"
    # LRU cap for `embed_one` query reuse. ONNX inference dominates recall
    # latency (~30–50 ms); a hit collapses that to a dict lookup. 128 keeps
    # memory bounded (~200 KB at 384 float32 per entry).
    _QUERY_CACHE_MAX: int = 128
    # Iter 31: drop the ONNX runtime after this many seconds of no embed
    # calls. Keeps daemon RSS low when nobody is actively recalling. Cold
    # reload on the next call is ~200 ms (already in tempfile-resistant
    # ~/.cache/fastembed thanks to iter 30). Override via env var.
    _IDLE_UNLOAD_SECONDS: int = 600

    def __init__(self, model_name: Optional[str] = None) -> None:
        if model_name:
            self.model = model_name
        # Iter 28: cheap __init__ — fail fast on missing package, but defer
        # the ONNX load (200 ms cached, several seconds cold) until the
        # first embed() call. Lets the daemon answer /health and the CLI
        # answer `skein up` long before the model is in memory.
        try:
            from fastembed import TextEmbedding  # noqa: F401  # type: ignore
        except ImportError as e:
            raise ImportError(
                "fastembed package is required for the FastembedProvider. "
                "Install it: pip install skein[fastembed]  (or: pip install fastembed)"
            ) from e
        self._model = None  # built on first embed call
        # Iter 31: monotonic timestamp of the most recent embed call.
        # idle_check_and_unload() uses this to decide when to drop the
        # ONNX runtime — see _IDLE_UNLOAD_SECONDS.
        import time
        self._last_call_at: float = 0.0
        self._monotonic = time.monotonic
        from collections import OrderedDict
        self._query_cache: "OrderedDict[str, list[float]]" = OrderedDict()

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # type: ignore
            # Iter 30: pin the cache dir to ~/.cache/fastembed so macOS's
            # /var/folders/* temp cleanup (purges files unused for >3 days)
            # can't silently delete the ONNX weights. The previous default
            # landed under tempfile.gettempdir() = /var/folders/<hash>/T/
            # and the model got reaped between sessions, then every embed
            # call silently failed with "File doesn't exist" and the
            # provider quietly fell back to BM25-only — making recall
            # return zero high-signal matches and the iter-29 empty-recall
            # fallback fire instead of real semantic results.
            import os
            from pathlib import Path
            cache_dir = Path(
                os.environ.get("SKEIN_FASTEMBED_CACHE")
                or (Path.home() / ".cache" / "fastembed")
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.model, cache_dir=str(cache_dir),
            )
        self._last_call_at = self._monotonic()
        return self._model

    def idle_check_and_unload(self) -> bool:
        """Drop the ONNX runtime if it's been idle for ``_IDLE_UNLOAD_SECONDS``.

        Returns True iff the model was actually unloaded. Called by a
        daemon background loop every minute. Reloads on the next embed
        call (~200 ms cold) — acceptable trade for ~200 MB of resident
        memory during inactive periods.

        Pinned by the LRU cache: a high-traffic embed_one query that
        keeps the LRU warm still goes through ``_ensure_model``, so the
        timer advances on every real call. Idle means truly idle.
        """
        import os
        # Allow override (tests + power users)
        try:
            window = float(
                os.environ.get("SKEIN_FASTEMBED_IDLE_SECONDS",
                               str(self._IDLE_UNLOAD_SECONDS)),
            )
        except (TypeError, ValueError):
            window = float(self._IDLE_UNLOAD_SECONDS)
        if self._model is None:
            return False
        if self._last_call_at == 0.0:
            # Loaded but never used? Refuse to unload — usually means a
            # warmup task is mid-flight on another thread.
            return False
        if (self._monotonic() - self._last_call_at) < window:
            return False
        logger.info(
            "FastembedProvider idle for %ds — unloading ONNX runtime to free RAM",
            int(self._monotonic() - self._last_call_at),
        )
        self._model = None
        return True

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        return [vec.tolist() for vec in model.embed(list(texts))]

    def embed_one(self, text: str) -> list[float]:
        cache = self._query_cache
        cached = cache.get(text)
        if cached is not None:
            cache.move_to_end(text)
            return cached
        model = self._ensure_model()
        vec = next(iter(model.embed([text]))).tolist()
        cache[text] = vec
        if len(cache) > self._QUERY_CACHE_MAX:
            cache.popitem(last=False)
        return vec


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small (1536-dim)."""

    dimension: int = 1536
    model: str = "text-embedding-3-small"
    is_real: bool = True
    name: str = "openai"

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install it: pip install openai"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# One-time deprecation warning flag for the "gemini" alias.
_gemini_alias_warned = False


def get_provider(name: str) -> EmbeddingProvider:
    """Return an embedding provider by name.

    Valid names: ``fastembed`` (default, local), ``openai`` (cloud),
    ``bm25`` (no-op vector, FTS5-only), ``hash`` (tests only).

    The literal ``"gemini"`` is accepted as a deprecated alias for
    ``"fastembed"`` so configs written before iter 27 don't crash the
    daemon on upgrade. A one-time warning is logged.
    """
    global _gemini_alias_warned
    name = name.lower().strip()
    if name == "gemini":
        if not _gemini_alias_warned:
            logger.warning(
                "Config has embedding_provider='gemini' — the Gemini "
                "embedding API was removed in iter 27. Aliasing to "
                "'fastembed' (local, 384-dim). Run `skein up` to migrate "
                "your config and re-ingest old fragments."
            )
            _gemini_alias_warned = True
        return FastembedProvider()
    if name in ("bm25", "none", "off"):
        return BM25OnlyProvider()
    if name == "hash":
        return HashEmbeddingProvider()
    if name == "fastembed":
        return FastembedProvider()
    if name == "openai":
        return OpenAIEmbeddingProvider()
    raise ValueError(
        f"Unknown embedding provider '{name}'. "
        "Valid options: fastembed, openai, bm25, hash"
    )


def best_available_provider_name() -> str:
    """Pick the default provider for new installs.

    Always ``fastembed`` — local, zero-config, no API key. Users who want
    cloud SOTA can opt into ``openai`` explicitly via
    ``skein config set embedding_provider openai``.
    """
    return "fastembed"


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity between two 1-D float32 arrays."""
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def vec_to_bytes(vec: list[float]) -> bytes:
    """Serialize a float list to raw float32 bytes for SQLite BLOB storage."""
    import numpy as np
    return np.array(vec, dtype=np.float32).tobytes()


def bytes_to_vec(raw: bytes, dimension: int) -> "np.ndarray":
    """Deserialize float32 bytes back to a numpy array."""
    import numpy as np
    arr = np.frombuffer(raw, dtype=np.float32)
    if len(arr) != dimension:
        # Dimension mismatch — return zeros (graceful fallback)
        return np.zeros(dimension, dtype=np.float32)
    return arr.copy()
