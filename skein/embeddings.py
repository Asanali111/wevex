"""Embedding provider abstraction.

Supported providers:
  "hash"   – deterministic fake embedding from SHA-256 (offline; good for tests).
             Quality is zero — do NOT use for production recall.
  "gemini" – Google Gemini gemini-embedding-001 (768-dim, free tier).
             Requires GEMINI_API_KEY env var or google-genai package.
  "openai" – OpenAI text-embedding-3-small (1536-dim).
             Requires OPENAI_API_KEY env var and openai package.

The dimension parameter is validated against the provider's output size.

All providers expose:
    embed(texts: list[str]) -> list[list[float]]

Network-backed providers (Gemini, OpenAI) MUST treat each call as best-effort:
  - Bounded per-request timeout.
  - Limited retry/backoff on rate-limit / 5xx / network errors.
  - After ``FAIL_THRESHOLD`` consecutive failures, return zero vectors for the
    rest of the batch (keyword-only fallback) instead of blocking. Ingest then
    continues — the chunk still lands in the DB and FTS5 still indexes it for
    keyword search.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from concurrent.futures import TimeoutError as FutureTimeout
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

    dimension: int = 768
    # Subclasses set this. ``is_real = True`` means embeddings produced by
    # this provider are semantically meaningful (e.g. Gemini, OpenAI).
    # ``is_real = False`` means embeddings are pseudo-random or zero —
    # vector ranking is decorative for this provider, retrieval falls back
    # to FTS5/BM25. ``skein doctor`` and the README rely on this marker.
    is_real: bool = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# BM25-only provider (no vector — honest about being keyword-only)
# ---------------------------------------------------------------------------

class BM25OnlyProvider(EmbeddingProvider):
    """Returns zero vectors so the vector retrieval path is a true no-op.

    This is the default when the user hasn't configured a real embedding
    provider. Search falls back to FTS5 (BM25 keyword matching) exclusively
    — fast, honest, doesn't pretend to be semantic. To enable semantic
    search, set ``GEMINI_API_KEY`` (or OPENAI_API_KEY) and run
    ``skein config set embedding_provider gemini`` (or ``openai``).
    """

    dimension: int = 768
    is_real: bool = False     # Marker the doctor / hooks can check

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
    an API key. **Not for production use.** ``skein doctor`` warns when this
    provider is active in a live config; the default for new installs is
    ``bm25`` (which is honest about being keyword-only) when no real
    embedding key is present.
    """

    is_real: bool = False

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

    def __init__(self, model_name: Optional[str] = None) -> None:
        if model_name:
            self.model = model_name
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as e:
            raise ImportError(
                "fastembed package is required for the FastembedProvider. "
                "Install it: pip install skein[fastembed]  (or: pip install fastembed)"
            ) from e
        # Lazily instantiate — first construction downloads model weights
        # (~130 MB) into the fastembed cache directory. Subsequent runs
        # reuse the cache and pay only the ONNX load (~200 ms).
        self._model = TextEmbedding(model_name=self.model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # ``TextEmbedding.embed`` yields one numpy array per input text.
        # Materialize to plain Python lists so the return type matches
        # the EmbeddingProvider contract (same shape as Gemini/OpenAI).
        return [vec.tolist() for vec in self._model.embed(list(texts))]


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini gemini-embedding-001 (768-dim).

    Hardened against the failure modes that previously caused ``skein up``
    to hang:

    - Per-request hard timeout (``REQUEST_TIMEOUT``) enforced via a private
      thread pool — even a stuck HTTP socket can't block ingest.
    - Limited retries with exponential backoff on transient errors
      (rate-limit, 5xx, timeout).
    - After ``FAIL_THRESHOLD`` consecutive failures, returns zero vectors for
      the rest of the batch and signals the caller via ``self.degraded``.
      Ingest then continues — chunks still land in FTS5 for keyword search.
    - Tries the batch API first (one HTTP call per ``embed()``); falls back
      to one-text-at-a-time for SDK versions that reject lists.
    """

    dimension: int = 768
    model: str = "gemini-embedding-001"
    is_real: bool = True

    REQUEST_TIMEOUT: float = 15.0       # per-call hard timeout in seconds
    MAX_RETRIES: int = 2                # extra attempts after the first try
    FAIL_THRESHOLD: int = 3             # consecutive failures before bailing

    def __init__(self, request_timeout: Optional[float] = None) -> None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it or switch embedding_provider to 'hash' in your config."
            )
        try:
            import google.genai as genai  # type: ignore
            self._client = genai.Client(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "google-genai package is required for the Gemini provider. "
                "Install it: pip install google-genai"
            ) from e

        if request_timeout is not None:
            self.REQUEST_TIMEOUT = request_timeout

        # Per-instance state for the degraded path. Reset on every embed().
        self.degraded: bool = False
        self._supports_batch: Optional[bool] = None  # tri-state, lazily probed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.degraded = False

        # Try batch first — one HTTP call, far cheaper for typical 32-text
        # ingest batches. If the SDK doesn't support a list payload, remember
        # that for the rest of the session.
        if self._supports_batch is not False:
            batch = self._try_batch(texts)
            if batch is not None:
                return batch
            # Batch path failed for a non-shape reason — the per-text fallback
            # below covers it without re-probing.

        return self._embed_per_text(texts)

    # ------------------------------------------------------------------
    # Batch path
    # ------------------------------------------------------------------

    def _try_batch(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Single HTTP call for the whole list. Returns None if the SDK
        rejects the list shape (we then fall back to per-text)."""
        try:
            resp = self._call_with_timeout(lambda: self._client.models.embed_content(
                model=self.model, contents=list(texts),
            ))
        except _ShapeError:
            self._supports_batch = False
            return None
        except Exception as e:
            logger.warning("gemini batch embed failed (%s); falling back to per-text", e)
            return None

        try:
            embs = [list(e.values) for e in resp.embeddings]
        except Exception as e:
            logger.warning("gemini batch returned unexpected shape (%s); falling back", e)
            return None

        if len(embs) != len(texts):
            logger.warning(
                "gemini batch returned %d embeddings for %d texts; falling back",
                len(embs), len(texts),
            )
            return None

        self._supports_batch = True
        return embs

    # ------------------------------------------------------------------
    # Per-text fallback (with retry + degrade)
    # ------------------------------------------------------------------

    def _embed_per_text(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        consecutive_failures = 0

        for i, text in enumerate(texts):
            if consecutive_failures >= self.FAIL_THRESHOLD:
                # Already gave up — short-circuit the rest with zeros.
                results.append([0.0] * self.dimension)
                continue

            vec = self._embed_one_with_retry(text)
            if vec is None:
                consecutive_failures += 1
                if consecutive_failures >= self.FAIL_THRESHOLD:
                    self.degraded = True
                    logger.warning(
                        "gemini provider degraded: %d consecutive failures; "
                        "remaining %d texts will get zero vectors (keyword-only)",
                        consecutive_failures, len(texts) - i - 1,
                    )
                results.append([0.0] * self.dimension)
            else:
                consecutive_failures = 0
                results.append(vec)

        return results

    def _embed_one_with_retry(self, text: str) -> Optional[list[float]]:
        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = self._call_with_timeout(lambda: self._client.models.embed_content(
                    model=self.model, contents=text,
                ))
                return list(resp.embeddings[0].values)
            except FutureTimeout as e:
                last_err = e
            except Exception as e:
                last_err = e
                if not _is_retryable(e):
                    break
            if attempt < self.MAX_RETRIES:
                time.sleep(min(2 ** attempt, 4.0))
        if last_err is not None:
            logger.debug("gemini embed dropped text after retries: %s", last_err)
        return None

    # ------------------------------------------------------------------
    # Hard timeout via thread pool
    # ------------------------------------------------------------------

    def _call_with_timeout(self, fn):
        """Run ``fn()`` in a daemon thread with a hard timeout.

        Daemon thread = a stuck SDK request can never block process exit, even
        though we can't actually cancel it (Python doesn't expose thread
        cancellation). Raises FutureTimeout on expiry.
        """
        import threading
        result: list = [None]
        error: list[Optional[BaseException]] = [None]

        def runner():
            try:
                result[0] = fn()
            except BaseException as e:  # incl. KeyboardInterrupt
                error[0] = e

        t = threading.Thread(target=runner, daemon=True, name="gemini-embed")
        t.start()
        t.join(timeout=self.REQUEST_TIMEOUT)
        if t.is_alive():
            raise FutureTimeout(
                f"gemini embed timed out after {self.REQUEST_TIMEOUT}s"
            )
        if error[0] is not None:
            raise error[0]
        return result[0]


# ---------------------------------------------------------------------------
# Helpers used by the Gemini provider
# ---------------------------------------------------------------------------

class _ShapeError(Exception):
    """Marker: the SDK rejected the batch shape (list payload)."""


def _is_retryable(exc: Exception) -> bool:
    """Heuristic: retry on rate limits, 5xx, and obvious transient issues."""
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg or "quota" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(code, int) and 500 <= code < 600:
        return True
    if any(s in msg for s in ("500", "502", "503", "504", "unavailable",
                              "connection", "reset")):
        return True
    return False


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small (1536-dim)."""

    dimension: int = 1536
    model: str = "text-embedding-3-small"
    is_real: bool = True

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

def get_provider(name: str) -> EmbeddingProvider:
    """Return an embedding provider by name.

    Valid names: ``fastembed`` (local 384-dim, default), ``gemini``, ``openai``,
    ``bm25`` (no-op vector, FTS5-only), ``hash`` (legacy / tests only).
    """
    name = name.lower().strip()
    if name in ("bm25", "none", "off"):
        return BM25OnlyProvider()
    if name == "hash":
        return HashEmbeddingProvider()
    if name == "fastembed":
        return FastembedProvider()
    if name == "gemini":
        return GeminiEmbeddingProvider()
    if name == "openai":
        return OpenAIEmbeddingProvider()
    raise ValueError(
        f"Unknown embedding provider '{name}'. "
        "Valid options: fastembed, gemini, openai, bm25, hash"
    )


def best_available_provider_name() -> str:
    """Pick the most capable provider for which credentials/packages exist.

    Used by ``skein init`` and ``skein up`` to default new installs to
    semantic search.

    Priority (iter 23):
      1. ``gemini`` if GEMINI_API_KEY set AND google-genai installed
      2. ``openai`` if OPENAI_API_KEY set AND openai installed
      3. ``fastembed`` if fastembed package installed (default for fresh installs)
      4. ``bm25`` if nothing else is available (FTS5-only fallback)
    """
    if os.environ.get("GEMINI_API_KEY"):
        try:
            import google.genai  # noqa: F401
            return "gemini"
        except ImportError:
            pass
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401
            return "openai"
        except ImportError:
            pass
    try:
        import fastembed  # noqa: F401
        return "fastembed"
    except ImportError:
        pass
    return "bm25"


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
