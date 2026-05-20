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

    # Iter 28: provider construction is now cheap — FastembedProvider's
    # __init__ no longer loads the ONNX model; that happens on first embed
    # call. So we can still build it eagerly here without blocking /health.
    provider = get_provider(cfg.embedding_provider)
    set_provider(provider)
    logger.info("Embedding provider: %s (dim=%d)", cfg.embedding_provider, provider.dimension)

    # Iter 29 day-one: fire a throwaway embed in a worker thread so the
    # ONNX runtime (and the model itself, if first launch on this machine)
    # is hot by the time the first MCP recall arrives. /health stays
    # responsive because this is `create_task` + `to_thread` — pure
    # background work. Without this, a fresh user's first `recall` call
    # ate 7–8 s of ONNX cold-start and calibrated their trust as
    # "Skein is slow." Failures here must NOT crash the daemon; we log
    # them and let lazy-load handle the retry on the first real call.
    async def _warm_embedding_provider() -> None:
        try:
            await asyncio.to_thread(provider.embed_one, "warmup")
            logger.info("Embedding provider warm.")
        except Exception:
            logger.warning("Embedding warmup failed; first recall will lazy-load.",
                           exc_info=True)
    warmup_task = asyncio.create_task(_warm_embedding_provider())

    # Iter 23: warn loudly if the stored embeddings' dimension doesn't match
    # the active provider. Cosine similarity between a 384-dim query and a
    # 768-dim stored vector is undefined — recall results would be garbage
    # until the user re-embeds with `skein ingest . --reset`. We don't auto-
    # invalidate (irreversible); just log the situation clearly so it shows
    # up in `skein doctor` and the daemon stderr log.
    try:
        stored_dim = storage.peek_embedding_dimension()
        provider_dim = getattr(provider, "dimension", None)
        if stored_dim and provider_dim and stored_dim != provider_dim:
            logger.warning(
                "Embedding-provider mismatch: stored dim=%d, active provider=%r dim=%d. "
                "Existing embeddings are unsearchable until re-embedded. "
                "Run: skein ingest . --reset",
                stored_dim, cfg.embedding_provider, provider_dim,
            )
    except Exception:
        logger.debug("Embedding dimension peek failed; skipping mismatch check", exc_info=True)

    # Note: filesystem watchers do *not* run inside the daemon process.
    # On macOS, the daemon runs under launchd which has restricted TCC
    # access (no read access to ~/Documents/, ~/Desktop/, etc.). Watchers
    # are spawned as session-scoped subprocesses by `skein up` instead, so
    # they inherit the user's full filesystem access.

    # Background maintenance tasks
    task1 = asyncio.create_task(_lease_cleanup_loop(storage, cfg.lease_cleanup_interval))
    task2 = asyncio.create_task(_stale_mark_loop(storage, cfg.stale_mark_interval))
    # ADR-002 / iter 26: daemon-side replacements for deleted CLI commands.
    # Auto-sync owns AGENTS.md regen; auto-approve drains the inbox.
    task3 = asyncio.create_task(_agents_md_sync_loop(
        cfg.db_path, cfg.base_url, cfg.agents_md_sync_interval,
    ))
    task4 = asyncio.create_task(_inbox_auto_approve_loop(
        cfg.db_path, cfg, provider, cfg.inbox_auto_approve_interval,
    ))
    # Iter 28 boot-perf: the code/docs scanners used to run synchronously
    # from `skein up`, costing 2–4 s every warm boot. Daemon-side sweep
    # owns them now so the CLI can return as soon as the daemon is healthy.
    task5 = asyncio.create_task(_passive_scan_loop(
        cfg.db_path, cfg, provider, cfg.passive_scan_interval,
    ))

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
    task3.cancel()
    task4.cancel()
    task5.cancel()
    warmup_task.cancel()
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


async def _agents_md_sync_loop(db_path: str, daemon_url: str, interval: int) -> None:
    """ADR-002 / iter 26: replace `skein sync` with a daemon-side regen.

    For each registered project, render AGENTS.md, hash the bytes, and only
    write if the hash differs from what's on disk. Opens its own Storage
    handle per sweep so the loop is independent of the lifespan-managed
    primary connection.
    """
    from .agents_md import sync_agents_md_for_project
    from .projects import list_projects

    while True:
        await asyncio.sleep(interval)
        try:
            from pathlib import Path
            projects = list_projects()
        except Exception as e:
            logger.warning("agents_md sync: list_projects failed: %s", e)
            continue
        if not projects:
            continue
        try:
            sweep_storage = Storage(db_path)
        except Exception as e:
            logger.warning("agents_md sync: open storage failed: %s", e)
            continue
        try:
            for project in projects:
                try:
                    root = Path(project.root)
                    if not root.is_dir():
                        continue
                    sync_agents_md_for_project(
                        storage=sweep_storage,
                        scope_handle=project.scope,
                        repo_root=root,
                        daemon_url=daemon_url,
                    )
                except Exception as e:
                    logger.warning(
                        "agents_md sync failed for %s: %s", project.scope, e,
                    )
        finally:
            sweep_storage.close()


async def _inbox_auto_approve_loop(db_path: str, cfg, provider, interval: int) -> None:
    """ADR-002 / iter 26: replace `skein inbox auto-approve` with a daemon
    sweep. Anything above the confidence threshold gets promoted to a real
    fragment; anything older than ``inbox_auto_reject_days`` that's still
    pending gets marked rejected so the queue self-drains.
    """
    from datetime import datetime, timedelta, timezone

    from .embeddings import vec_to_bytes
    from .models import CommitCreate, FragmentCreate, IdentityCreate
    from .auth import token_prefix as _tp

    while True:
        await asyncio.sleep(interval)
        try:
            sweep_storage = Storage(db_path)
        except Exception as e:
            logger.warning("inbox auto-approve: open storage failed: %s", e)
            continue
        try:
            candidates = sweep_storage.list_extraction_candidates(limit=500)
            if not candidates:
                continue
            owner = sweep_storage.get_or_create_identity(IdentityCreate(
                handle=f"user:{_tp(cfg.bearer_token)}",
                type="user", name="local-user",
            ))
            promote_threshold = cfg.inbox_auto_approve_threshold
            reject_cutoff = datetime.now(timezone.utc) - timedelta(
                days=cfg.inbox_auto_reject_days,
            )
            promoted = 0
            rejected = 0
            for c in candidates:
                created_raw = c.get("created_at") or ""
                normalised = created_raw.replace(" ", "T", 1).replace("Z", "+00:00")
                try:
                    ts = datetime.fromisoformat(normalised)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    ts = None
                # Promote high-confidence
                if c["confidence"] >= promote_threshold:
                    try:
                        commit = sweep_storage.create_commit(CommitCreate(
                            author_id=owner.id, scope_id=c["scope_id"],
                            message=f"[auto-approve] {c['content'][:60]}",
                        ))
                        import json as _json
                        emb_bytes = None
                        try:
                            vec = provider.embed_one(c["content"])
                            emb_bytes = vec_to_bytes(vec)
                        except Exception:
                            pass
                        frag = sweep_storage.create_fragment(
                            FragmentCreate(
                                content=c["content"], type=c["type"],
                                scope_id=c["scope_id"], owner_id=owner.id,
                                territory=c.get("territory"),
                                tags=_json.loads(c.get("tags") or "[]"),
                                created_by_tool=c["source_tool"],
                                extraction_method=c["source_tool"],
                                extraction_confidence=c["confidence"],
                                metadata={"promoted_via": "inbox-auto-approve"},
                            ),
                            commit_id=commit.id, embedding=emb_bytes,
                        )
                        sweep_storage.mark_candidate_status(
                            c["id"], "approved", promoted_fragment_id=frag.id,
                        )
                        promoted += 1
                    except Exception as e:
                        logger.warning(
                            "inbox auto-approve: promote %s failed: %s",
                            c["id"][:8], e,
                        )
                # Reject anything too old that didn't clear the threshold.
                elif ts is not None and ts < reject_cutoff:
                    if sweep_storage.mark_candidate_status(c["id"], "rejected"):
                        rejected += 1
            if promoted or rejected:
                logger.info(
                    "inbox sweep: promoted=%d rejected=%d", promoted, rejected,
                )
        except Exception as e:
            logger.warning("inbox auto-approve loop error: %s", e)
        finally:
            sweep_storage.close()


async def _passive_scan_loop(db_path: str, cfg, provider, interval: int) -> None:
    """Iter 28 boot-perf: own the package-manifest scan + docs scan on the
    daemon side so ``skein up`` doesn't pay the cost on every invocation.

    Walks every registered project's `scan_project()` (package.json,
    pyproject.toml, Dockerfile, CI, etc.) and `scan_docs()` (README,
    CHANGELOG, ADRs) outputs through ``promote_scanned_facts``. The
    `_agents_md_sync_loop` then picks up any new facts and regenerates
    AGENTS.md within its 60-s interval. Net behaviour vs. the old CLI
    blocks is identical apart from the up-to-``interval``-seconds delay
    on freshly added projects — by then the user is already coding.

    First iteration sleeps ``min(interval, 5)`` s so daemon boot is
    never blocked by a scanner walk. Sweep failures are logged and the
    loop continues — never wedges the daemon.
    """
    from pathlib import Path
    from .auth import token_prefix as _tp
    from .docs_watcher import scan_docs
    from .models import IdentityCreate
    from .passive import promote_scanned_facts
    from .projects import list_projects
    from .scanner import scan_project

    # Iter 29 day-one: 1 s stagger is enough to let uvicorn bind the port
    # and /health become live; previously 5 s, which meant fresh users got
    # an empty `recall` for 5+ s after `skein up` returned. Cold-start
    # corpus (docs scan, scanner facts) lands inside the first second now.
    await asyncio.sleep(min(interval, 1))
    while True:
        try:
            projects = list_projects()
        except Exception as e:
            logger.warning("passive scan: list_projects failed: %s", e)
            projects = []
        for project in projects:
            try:
                root = Path(project.root)
                if not root.is_dir():
                    continue
                sweep_storage = Storage(db_path)
                try:
                    scope_obj = sweep_storage.get_scope(project.scope)
                    if not scope_obj:
                        continue
                    owner = sweep_storage.get_or_create_identity(IdentityCreate(
                        handle=f"user:{_tp(cfg.bearer_token)}",
                        type="user", name="local-user",
                    ))
                    facts = scan_project(root)
                    if facts:
                        promote_scanned_facts(
                            facts, storage=sweep_storage, provider=provider,
                            scope_id=scope_obj.id, owner_id=owner.id,
                            source_tool="code-scanner",
                        )
                    doc_facts = scan_docs(root)
                    if doc_facts:
                        promote_scanned_facts(
                            doc_facts, storage=sweep_storage, provider=provider,
                            scope_id=scope_obj.id, owner_id=owner.id,
                            source_tool="docs-scanner",
                        )
                finally:
                    sweep_storage.close()
            except Exception as e:
                logger.warning(
                    "passive scan failed for %s: %s", project.scope, e,
                )
        await asyncio.sleep(interval)
