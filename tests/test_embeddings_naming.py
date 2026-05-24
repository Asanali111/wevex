"""Guard against the gemini-embedding / gemini-CLI naming confusion.

Two unrelated things in Skein used to share the prefix "gemini":

  1. The **Gemini embedding API** — removed in iter 27 because its rate
     limits wedged the daemon's event loop. The string "gemini" is now
     only kept as a deprecated alias mapping to FastembedProvider so
     existing on-disk configs don't crash the daemon on upgrade.

  2. The **Gemini CLI** as an LLM client — a fully-supported sync target
     that's identified everywhere by ``"gemini_cli"`` (underscore), not
     ``"gemini"``. Skein writes an MCP config snippet so the ``gemini``
     binary can connect to the local daemon, the same way it does for
     Claude Code, Cursor, Codex, Antigravity, etc.

These tests pin both invariants so a future refactor can't silently
reintroduce a "gemini" embedding provider, or rename the Gemini CLI
client to plain "gemini" and collide with the embedding alias.
"""
from __future__ import annotations

import pytest


def test_gemini_string_is_deprecated_alias_for_fastembed():
    """A user config still naming ``embedding_provider='gemini'`` must keep
    working — the daemon would otherwise crash on launchd respawn after a
    pip upgrade. ``get_provider('gemini')`` returns a FastembedProvider."""
    from skein.embeddings import FastembedProvider, get_provider

    provider = get_provider("gemini")
    assert isinstance(provider, FastembedProvider), (
        "'gemini' must alias to FastembedProvider so legacy configs don't "
        "crash. Do NOT reintroduce a separate GeminiEmbeddingProvider — "
        "the API rate-limits wedged the daemon's asyncio event loop."
    )


def test_gemini_alias_normalised_at_config_load():
    """SkeinConfig must normalise the deprecated alias in __init__ so all
    downstream callers (server.py, hooks.py, …) only ever see 'fastembed'."""
    from skein.config import SkeinConfig

    cfg = SkeinConfig({"embedding_provider": "gemini"})
    assert cfg.embedding_provider == "fastembed"


def test_gemini_alias_load_config_writes_back(tmp_path, monkeypatch):
    """A persisted config still naming 'gemini' must be rewritten to
    'fastembed' on read — keeps the file matching reality and prevents
    the daemon from log-spamming the deprecation warning forever."""
    import json
    from skein.config import load_config

    # Conftest pins SKEIN_EMBEDDING_PROVIDER=hash for offline tests; the
    # production scenario being tested here is "no env override, legacy
    # disk value", so drop the override before calling.
    monkeypatch.delenv("SKEIN_EMBEDDING_PROVIDER", raising=False)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "embedding_provider": "gemini",
        "embedding_dimension": 768,   # legacy key — must be stripped
        "bearer_token": "x" * 64,
    }))

    cfg = load_config(cfg_path)
    assert cfg.embedding_provider == "fastembed"

    # File should be rewritten without 'gemini' and without embedding_dimension.
    persisted = json.loads(cfg_path.read_text())
    assert persisted["embedding_provider"] == "fastembed"
    assert "embedding_dimension" not in persisted


def test_no_separate_gemini_embedding_class():
    """A future refactor must not reintroduce a Gemini *embedding* provider.

    If you genuinely need cloud embeddings, extend OpenAIEmbeddingProvider
    or add a new differently-named class. Bringing back a Gemini-API-backed
    embedding provider regresses the iter 27 fix where its rate limits
    wedged the daemon and produced 90-second `skein up` hangs.
    """
    import skein.embeddings as emb

    forbidden = [
        name for name in dir(emb)
        if name.startswith("Gemini") and "Embedding" in name
    ]
    assert forbidden == [], (
        f"Found resurrected Gemini embedding class(es): {forbidden}. "
        "See skein/embeddings.py module docstring for why this is banned."
    )


def test_get_provider_rejects_unknown_names():
    """Sanity: factory rejects strings that aren't in the supported set."""
    from skein.embeddings import get_provider

    with pytest.raises(ValueError):
        get_provider("nope")


def test_fastembed_provider_dimension_and_identity():
    """FastembedProvider declares 384-dim BGE-small with is_real=True and
    name='fastembed'. Used by server.py / doctor for capability checks
    without instantiating the heavy ONNX runtime."""
    from skein.embeddings import FastembedProvider

    assert FastembedProvider.dimension == 384
    assert FastembedProvider.is_real is True
    assert FastembedProvider.name == "fastembed"
    # Canonical names the model attribute ``model`` (not ``model_name``).
    assert "BAAI/bge-small" in FastembedProvider.model


def test_gemini_cli_client_id_uses_underscore():
    """The Gemini CLI LLM client identifier MUST stay 'gemini_cli'.

    Renaming it to plain 'gemini' would collide with the embedding-alias
    namespace and the next person debugging "why is my Skein daemon
    embedding via the Gemini CLI" would lose an afternoon.
    """
    from skein.clients import GeminiCLIClient

    assert GeminiCLIClient.id == "gemini_cli"
    assert "cli" in GeminiCLIClient.display_name.lower()
