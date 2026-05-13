"""Shared pytest fixtures."""
from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Force hash embedding so tests work offline
os.environ["SKEIN_EMBEDDING_PROVIDER"] = "hash"


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Return a path to a fresh temp SQLite DB (auto-deleted)."""
    return str(tmp_path / "test_skein.db")


@pytest.fixture
def storage(tmp_db: str):
    """A fresh Storage instance backed by a temp DB."""
    from skein.storage import Storage
    s = Storage(tmp_db)
    yield s
    s.close()


@pytest.fixture
def provider():
    """Hash embedding provider (offline, deterministic)."""
    from skein.embeddings import HashEmbeddingProvider
    return HashEmbeddingProvider()


@pytest.fixture
def seeded_storage(storage):
    """Storage with a default user identity and scope pre-created."""
    from skein.models import IdentityCreate, ScopeCreate
    user = storage.create_identity(IdentityCreate(
        handle="user:testuser", type="user", name="Test User",
    ))
    scope = storage.create_scope(ScopeCreate(
        handle="project:test", type="project",
        name="Test Project", owner_id=user.id,
    ))
    storage._test_user = user
    storage._test_scope = scope
    return storage


# ---------------------------------------------------------------------------
# FastAPI TestClient with auth injected
# ---------------------------------------------------------------------------

TEST_TOKEN = "test-token-abcdef1234567890abcdef1234567890abcdef1234567890abcdef12"


@pytest.fixture
def app(tmp_db: str):
    """A fully configured FastAPI test app backed by a temp DB."""
    from skein.config import SkeinConfig, reset_config
    from skein.dependencies import set_provider, set_storage
    from skein.embeddings import HashEmbeddingProvider
    from skein.server import create_app
    from skein.storage import Storage

    cfg = SkeinConfig({
        "db_path": tmp_db,
        "bearer_token": TEST_TOKEN,
        "embedding_provider": "hash",
    })
    reset_config(cfg)

    storage = Storage(tmp_db)
    set_storage(storage)
    set_provider(HashEmbeddingProvider())

    application = create_app(cfg)

    yield application

    reset_config(None)
    storage.close()


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """TestClient with the auth header pre-set."""
    with TestClient(app, raise_server_exceptions=True) as c:
        c.headers["Authorization"] = f"Bearer {TEST_TOKEN}"
        yield c


@pytest.fixture
def authed_client(client: TestClient) -> TestClient:
    """Alias — same as client (all our tests use auth)."""
    return client
