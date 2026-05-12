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
import os
import time
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
from .routers.briefing import router as briefing_router
from .routers.chunks import router as chunks_router
from .routers.commits import router as commits_router
from .routers.fragments import router as fragments_router
from .routers.identities import router as identities_router
from .routers.leases import router as leases_router
from .routers.scopes import router as scopes_router
from .storage import Storage

logger = logging.getLogger("skein.server")


# ---------------------------------------------------------------------------
# Daemon uptime tracker (set in lifespan, read by /v1/briefing + MCP briefing)
# ---------------------------------------------------------------------------
_DAEMON_STARTED_AT: Optional[float] = None


def get_daemon_uptime_seconds() -> int:
    """Seconds since the daemon entered its lifespan. 0 if not started yet."""
    if _DAEMON_STARTED_AT is None:
        return 0
    return int(time.time() - _DAEMON_STARTED_AT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _DAEMON_STARTED_AT
    _DAEMON_STARTED_AT = time.time()
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

    # Common setup for the two passive watchers — both need their own SQLite
    # handle per poll and the local-user identity.
    from .models import IdentityCreate
    from .auth import token_prefix as _tp

    def _storage_factory():
        return Storage(cfg.db_path)

    def _get_owner(st):
        ident = st.get_or_create_identity(IdentityCreate(
            handle=f"user:{_tp(cfg.bearer_token)}",
            type="user", name="local-user",
        ))
        return ident.id

    # Git commit watcher (iter 15): the *primary* decision-capture path.
    # Every new commit in a Skein-up'd repo becomes a `decision` fragment
    # with the commit SHA as `created_against_commit`. Strictly better
    # signal than chat extraction — devs write commit messages on purpose.
    git_watcher = None
    try:
        from .git_watcher import MultiProjectGitWatcher
        git_watcher = MultiProjectGitWatcher(
            storage_factory=_storage_factory,
            provider=provider,
            get_owner_id=_get_owner,
            poll_interval=10.0,
        )
        git_watcher.start()
    except Exception:
        logger.exception("Git commit watcher failed to start; skipping.")

    # Transcript watcher (iter 14.2, demoted to opt-in in iter 15): tails
    # Claude Code JSONL transcripts. Off by default because it produces noise
    # the inbox has to filter; the git watcher is the better default path.
    # Enable with: SKEIN_TRANSCRIPT_WATCHER=1
    transcript_watcher = None
    if os.environ.get("SKEIN_TRANSCRIPT_WATCHER") == "1":
        try:
            from .transcript_watcher import MultiProjectTranscriptWatcher
            transcript_watcher = MultiProjectTranscriptWatcher(
                storage_factory=_storage_factory,
                provider=provider,
                poll_interval=3.0,
                get_owner_id=_get_owner,
            )
            transcript_watcher.start()
        except Exception:
            logger.exception("Transcript watcher failed to start; skipping.")

    yield

    task1.cancel()
    task2.cancel()
    for w in (git_watcher, transcript_watcher):
        if w is not None:
            try:
                w.stop(timeout=2.0)
            except Exception:
                pass
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
    app.include_router(briefing_router)
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
