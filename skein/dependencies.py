"""FastAPI dependency singletons: Storage, EmbeddingProvider.

Injected via Depends() in all routers. Tests replace them via
app.dependency_overrides.
"""
from __future__ import annotations

from typing import Optional

from .embeddings import EmbeddingProvider, get_provider
from .storage import Storage

# Module-level singletons (initialised by server.py on startup)
_storage: Optional[Storage] = None
_provider: Optional[EmbeddingProvider] = None


def set_storage(storage: Storage) -> None:
    global _storage
    _storage = storage


def set_provider(provider: EmbeddingProvider) -> None:
    global _provider
    _provider = provider


def get_storage() -> Storage:
    if _storage is None:
        raise RuntimeError("Storage not initialised. Call set_storage() first.")
    return _storage


def get_provider() -> EmbeddingProvider:
    if _provider is None:
        raise RuntimeError("EmbeddingProvider not initialised. Call set_provider() first.")
    return _provider
