"""REST router: /v1/chunks — codebase RAG (search + stats + delete-by-root).

Ingestion is intentionally CLI-only (`skein ingest`); we don't expose POST
ingest over HTTP because:
  • Walking a directory tree from inside a request is the wrong place to
    do bulk work (no progress feedback, blocks workers).
  • The CLI runs in the user's filesystem context; the daemon may run as
    a different user with no filesystem access.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import AuthContext, RequireAuth
from ..dependencies import get_provider, get_storage
from ..embeddings import EmbeddingProvider
from ..models import (
    Chunk,
    ChunkSearchRequest,
    ChunkSearchResponse,
    ChunkStats,
)
from ..retrieval import search_chunks
from ..storage import Storage

router = APIRouter(prefix="/v1/chunks", tags=["chunks"])


@router.post("/search", response_model=ChunkSearchResponse)
def search(
    req: ChunkSearchRequest,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
    provider: EmbeddingProvider = Depends(get_provider),
) -> ChunkSearchResponse:
    """Hybrid BM25 + vector + RRF over indexed code/document chunks."""
    return search_chunks(req, storage, provider)


@router.get("/search", response_model=ChunkSearchResponse)
def search_get(
    q: str = Query(..., min_length=1),
    scope: str = Query(...),
    languages: str | None = Query(None, description="Comma-separated language filter"),
    source_root: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
    provider: EmbeddingProvider = Depends(get_provider),
) -> ChunkSearchResponse:
    """GET variant — convenience for browser/curl usage."""
    lang_list = [l.strip() for l in languages.split(",")] if languages else None
    req = ChunkSearchRequest(
        query=q, scope=scope, languages=lang_list,
        source_root=source_root, limit=limit,
    )
    return search_chunks(req, storage, provider)


@router.get("", response_model=list[Chunk])
def list_chunks(
    scope: str = Query(..., description="Scope handle or ID"),
    source_root: str | None = None,
    language: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = 0,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> list[Chunk]:
    scope_obj = storage.get_scope(scope)
    if not scope_obj:
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    return storage.list_chunks(
        scope_id=scope_obj.id,
        source_root=source_root,
        language=language,
        limit=limit, offset=offset,
    )


@router.get("/stats", response_model=ChunkStats)
def stats(
    scope: str = Query(...),
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> ChunkStats:
    scope_obj = storage.get_scope(scope)
    if not scope_obj:
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    s = storage.chunk_stats(scope_id=scope_obj.id)
    return ChunkStats(
        scope=scope,
        total_chunks=s["total_chunks"],
        total_files=s["total_files"],
        by_language=s["by_language"],
        by_root=s["by_root"],
    )


@router.delete("/{source_root}", status_code=204)
def delete_root(
    source_root: str,
    scope: str = Query(...),
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> None:
    """Delete every chunk under a given source_root within a scope."""
    scope_obj = storage.get_scope(scope)
    if not scope_obj:
        raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
    storage.delete_chunks_by_root(scope_obj.id, source_root)
