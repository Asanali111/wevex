"""FastAPI application factory for the Skein daemon.

One process, one port (default 8765):
  /health        — public health check
  /v1/...        — REST API (auth required)
  /mcp           — MCP JSON-RPC (auth required)

Background tasks:
  - Every 60s:  clean up expired leases
  - Every hour: mark expired fragments stale
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import SkeinConfig, get_config
from .dependencies import set_provider, set_storage
from .embeddings import get_provider
from .mcp import router as mcp_router
from .models import HealthResponse
from .routers.chunks import router as chunks_router
from .routers.commits import router as commits_router
from .routers.fragments import router as fragments_router
from .routers.identities import router as identities_router
from .routers.leases import router as leases_router
from .routers.scopes import router as scopes_router
from .storage import Storage

logger = logging.getLogger("skein.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: SkeinConfig = app.state.cfg  # type: ignore[attr-defined]

    # Initialise storage
    storage = Storage(cfg.db_path)
    set_storage(storage)
    logger.info("Storage initialised at %s", cfg.db_path)

    # Initialise embedding provider
    provider = get_provider(cfg.embedding_provider)
    set_provider(provider)
    logger.info("Embedding provider: %s (dim=%d)", cfg.embedding_provider, provider.dimension)

    # Note: filesystem watchers do *not* run inside the daemon process.
    # On macOS, the daemon runs under launchd which has restricted TCC
    # access (no read access to ~/Documents/, ~/Desktop/, etc.). Watchers
    # are spawned as session-scoped subprocesses by `skein up` instead, so
    # they inherit the user's full filesystem access.

    # Background maintenance tasks
    task1 = asyncio.create_task(_lease_cleanup_loop(storage, cfg.lease_cleanup_interval))
    task2 = asyncio.create_task(_stale_mark_loop(storage, cfg.stale_mark_interval))

    yield

    task1.cancel()
    task2.cancel()
    storage.close()
    logger.info("Storage closed.")


def create_app(cfg: Optional[SkeinConfig] = None) -> FastAPI:
    if cfg is None:
        cfg = get_config()

    app = FastAPI(
        title="Skein",
        description=(
            "Local MCP context bus for coding LLMs. "
            "Connects Claude Code, Cursor, Codex, Gemini CLI, Antigravity, "
            "Copilot, VS Code, and opencode to shared, typed context."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.cfg = cfg

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                       "http://localhost:3000", f"http://{cfg.host}:{cfg.port}"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(identities_router)
    app.include_router(scopes_router)
    app.include_router(fragments_router)
    app.include_router(commits_router)
    app.include_router(leases_router)
    app.include_router(chunks_router)
    app.include_router(mcp_router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        from .dependencies import get_storage as gs
        try:
            storage = gs()
            stats = storage.stats()
            return HealthResponse(
                status="ok",
                db_path=storage.db_path,
                fragment_count=stats["fragments"],
                scope_count=stats["scopes"],
                identity_count=stats["identities"],
            )
        except RuntimeError:
            return HealthResponse(status="starting")

    return app


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------

async def _lease_cleanup_loop(storage: Storage, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            storage.cleanup_expired_leases()
        except Exception as e:
            logger.warning("Lease cleanup error: %s", e)


async def _stale_mark_loop(storage: Storage, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            storage.mark_expired_fragments_stale()
        except Exception as e:
            logger.warning("Stale-mark error: %s", e)
