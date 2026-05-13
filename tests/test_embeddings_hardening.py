"""Tests for the iteration-10 GeminiEmbeddingProvider hardening.

We don't talk to the real Gemini API — instead we monkeypatch the SDK client
so we can simulate timeouts, rate limits, batch shapes, and persistent failure.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from skein import embeddings as emb

# ---------------------------------------------------------------------------
# Fake SDK client
# ---------------------------------------------------------------------------

class _FakeEmbeddings(SimpleNamespace):
    def __init__(self, vec):
        super().__init__(values=vec)


class _FakeResponse(SimpleNamespace):
    def __init__(self, embeddings):
        super().__init__(embeddings=embeddings)


class _FakeClient:
    """Drop-in for genai.Client. Configurable per-test."""

    def __init__(self, *, batch_supported=True, fail_n=0, fail_kind="rate",
                 hang=False, hang_for=999):
        self.batch_supported = batch_supported
        self.fail_n = fail_n        # how many initial calls to fail
        self.fail_kind = fail_kind  # 'rate' | 'shape' | 'fatal'
        self.hang = hang
        self.hang_for = hang_for
        self.calls = 0
        self.models = self  # client.models.embed_content -> self.embed_content

    def embed_content(self, *, model: str, contents):
        self.calls += 1
        if self.hang:
            time.sleep(self.hang_for)
        if self.calls <= self.fail_n:
            if self.fail_kind == "rate":
                raise RuntimeError("HTTP 429: rate limit exceeded")
            if self.fail_kind == "shape" and isinstance(contents, list):
                raise emb._ShapeError("list payload not supported")
            if self.fail_kind == "fatal":
                raise RuntimeError("400: invalid argument")
        if isinstance(contents, list):
            return _FakeResponse([_FakeEmbeddings([0.1 * i] * 768)
                                  for i in range(len(contents))])
        return _FakeResponse([_FakeEmbeddings([0.42] * 768)])


@pytest.fixture
def patched_provider(monkeypatch):
    """Build a provider whose internal client is a controllable fake."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # Stub the import-of-google-genai inside __init__
    monkeypatch.setattr(emb.GeminiEmbeddingProvider, "__init__",
                        _short_init, raising=True)

    def _make(**fake_kw):
        p = emb.GeminiEmbeddingProvider()
        p._client = _FakeClient(**fake_kw)
        # Speed up retries in tests
        p.MAX_RETRIES = 1
        p.REQUEST_TIMEOUT = 0.5
        p.FAIL_THRESHOLD = 2
        p.degraded = False
        return p
    return _make


def _short_init(self, request_timeout=None):
    """Replacement __init__ that skips the SDK import + executor setup."""
    self.degraded = False
    self._supports_batch = None
    self._client = None
    if request_timeout is not None:
        self.REQUEST_TIMEOUT = request_timeout


# ---------------------------------------------------------------------------
# Batch happy path
# ---------------------------------------------------------------------------

class TestBatch:
    def test_one_call_for_n_texts(self, patched_provider):
        p = patched_provider(batch_supported=True)
        out = p.embed(["a", "b", "c"])
        assert len(out) == 3
        assert p._client.calls == 1
        assert p._supports_batch is True
        assert p.degraded is False

    def test_falls_back_when_batch_returns_wrong_count(self, patched_provider):
        p = patched_provider()
        # Force the fake to return one-too-few embeddings
        original = p._client.embed_content

        def shrink(*args, **kw):
            resp = original(*args, **kw)
            if len(resp.embeddings) > 1:
                resp.embeddings = resp.embeddings[:-1]  # drop the last
            return resp
        p._client.embed_content = shrink
        out = p.embed(["a", "b", "c"])
        # Falls back to per-text — gets 3 embeddings.
        assert len(out) == 3


# ---------------------------------------------------------------------------
# Per-text fallback (when batch shape isn't supported)
# ---------------------------------------------------------------------------

class TestShapeFallback:
    def test_remembers_no_batch_for_session(self, patched_provider):
        p = patched_provider(fail_n=1, fail_kind="shape")
        # First call (batch) raises _ShapeError → remembered
        out = p.embed(["a", "b"])
        assert len(out) == 2
        assert p._supports_batch is False
        # Subsequent embed() goes straight to per-text — counts: 1 shape attempt
        # + 2 per-text successes = 3 total calls so far.
        assert p._client.calls == 3
        # Reset call counter and embed again — should NOT try the batch path.
        p._client.calls = 0
        p.embed(["x", "y"])
        assert p._client.calls == 2  # purely per-text


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_per_request_timeout(self, patched_provider):
        p = patched_provider(hang=True, hang_for=999)
        p.REQUEST_TIMEOUT = 0.2
        p.MAX_RETRIES = 0
        p.FAIL_THRESHOLD = 1
        start = time.monotonic()
        out = p.embed(["just one"])
        elapsed = time.monotonic() - start
        # Must return quickly — well under 1s even though the fake hangs forever.
        assert elapsed < 2.0
        # Returned a zero vector + degraded flag set.
        assert out == [[0.0] * 768]
        assert p.degraded is True


# ---------------------------------------------------------------------------
# Retry on rate limits
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retries_on_rate_limit_then_succeeds(self, patched_provider):
        # First batch attempt raises 429; per-text fallback succeeds.
        p = patched_provider(fail_n=1, fail_kind="rate")
        out = p.embed(["a", "b"])
        # 1 failed batch + 2 per-text successes = 3 calls
        assert len(out) == 2
        assert p._client.calls == 3

    def test_does_not_retry_on_fatal(self, patched_provider):
        # Force per-text mode with shape-not-supported
        p = patched_provider(fail_n=1, fail_kind="shape")
        p.embed(["bootstrap"])  # establishes _supports_batch=False
        # Now force a fatal error — should NOT retry
        p._client.fail_n = 1
        p._client.fail_kind = "fatal"
        p._client.calls = 0
        out = p.embed(["x"])
        assert p._client.calls == 1  # one attempt, no retries
        # Got zero vector
        assert out == [[0.0] * 768]


# ---------------------------------------------------------------------------
# Degrade after persistent failure
# ---------------------------------------------------------------------------

class TestDegrade:
    def test_short_circuits_after_threshold(self, patched_provider):
        # Force per-text mode
        p = patched_provider(fail_n=1, fail_kind="shape")
        p.embed(["init"])
        # Now make every call fail
        p._client.fail_n = 999
        p._client.fail_kind = "rate"  # retryable, so we hit MAX_RETRIES
        p._client.calls = 0
        p.MAX_RETRIES = 0
        p.FAIL_THRESHOLD = 2

        out = p.embed(["a", "b", "c", "d", "e"])
        assert len(out) == 5
        # All zero vectors
        assert all(v == [0.0] * 768 for v in out)
        # Should have stopped calling after FAIL_THRESHOLD failures
        # (2 attempts; remaining 3 short-circuit)
        assert p._client.calls == 2
        assert p.degraded is True


# ---------------------------------------------------------------------------
# _is_retryable helper
# ---------------------------------------------------------------------------

class TestIsRetryable:
    @pytest.mark.parametrize("msg", [
        "HTTP 429: too many requests",
        "rate limit exceeded",
        "quota exceeded",
        "request timed out",
        "connection reset",
        "503 service unavailable",
    ])
    def test_retryable_messages(self, msg):
        assert emb._is_retryable(RuntimeError(msg)) is True

    @pytest.mark.parametrize("msg", [
        "400 bad request",
        "401 unauthorized",
        "invalid argument",
    ])
    def test_non_retryable_messages(self, msg):
        assert emb._is_retryable(RuntimeError(msg)) is False

    def test_status_code_attr(self):
        e = RuntimeError("boom")
        e.status_code = 503
        assert emb._is_retryable(e) is True
