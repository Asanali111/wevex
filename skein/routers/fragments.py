"""REST router: /v1/fragments — CRUD + recall (hybrid search)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import AuthContext, RequireAuth
from ..dependencies import get_provider, get_storage
from ..embeddings import EmbeddingProvider, vec_to_bytes
from ..models import (
    Fragment, FragmentCreate, FragmentUpdate,
    RecallRequest, RecallResponse,
)
from ..retrieval import recall
from ..storage import ConflictError, Storage

router = APIRouter(prefix="/v1/fragments", tags=["fragments"])


@router.post("", response_model=Fragment, status_code=status.HTTP_201_CREATED)
def create_fragment(
    data: FragmentCreate,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
    provider: EmbeddingProvider = Depends(get_provider),
) -> Fragment:
    """Create a fragment.  Automatically embeds content and creates a commit."""
    # Validate scope exists
    scope = storage.get_scope(data.scope_id)
    if not scope:
        raise HTTPException(status_code=404, detail=f"Scope '{data.scope_id}' not found. "
                            "Create it first with POST /v1/scopes.")

    # Resolve scope handle → UUID, and force owner to authenticated user
    data.scope_id = scope.id
    data.owner_id = auth.user_id

    # Embed in the background (best-effort; fragment is created even if embedding fails)
    embedding_bytes: Optional[bytes] = None
    try:
        vec = provider.embed_one(data.content)
        embedding_bytes = vec_to_bytes(vec)
    except Exception:
        pass

    # Create commit first
    from ..models import CommitCreate
    commit = storage.create_commit(CommitCreate(
        author_id=auth.user_id,
        scope_id=scope.id,
        message=f"add {data.type}: {data.content[:60]}{'…' if len(data.content) > 60 else ''}",
        metadata={"agent_id": auth.agent_id},
    ))

    frag = storage.create_fragment(data, commit_id=commit.id, embedding=embedding_bytes)

    # Back-fill commit's fragments_added
    storage._conn.execute(
        "UPDATE commits SET fragments_added = ? WHERE id = ?",
        (f'["{frag.id}"]', commit.id),
    )

    return frag


@router.get("", response_model=list[Fragment])
def list_fragments(
    scope: Optional[str] = Query(None, description="Scope handle or ID"),
    type: Optional[str] = Query(None, description="Filter by fragment type"),
    include_stale: bool = False,
    limit: int = Query(50, ge=1, le=200),
    offset: int = 0,
    since: Optional[str] = Query(None, description="ISO 8601 timestamp; only return fragments created after this time"),
    exclude_tool: Optional[str] = Query(None, description="Exclude fragments whose created_by_tool equals this value"),
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> list[Fragment]:
    scope_id: Optional[str] = None
    if scope:
        scope_obj = storage.get_scope(scope)
        if not scope_obj:
            raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
        scope_id = scope_obj.id

    return storage.list_fragments(
        scope_id=scope_id, type_filter=type,
        include_stale=include_stale, limit=limit, offset=offset,
        since=since, exclude_tool=exclude_tool,
    )


@router.get("/search", response_model=RecallResponse)
def search_fragments(
    q: str = Query(..., min_length=1, description="Search query"),
    scope: str = Query(..., description="Scope handle to search within"),
    types: Optional[str] = Query(None, description="Comma-separated fragment types"),
    territory: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    include_stale: bool = False,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
    provider: EmbeddingProvider = Depends(get_provider),
) -> RecallResponse:
    type_list = [t.strip() for t in types.split(",")] if types else None
    req = RecallRequest(
        query=q, scope=scope, types=type_list,
        territory=territory, limit=limit, include_stale=include_stale,
    )
    return recall(req, storage, provider)


@router.post("/recall", response_model=RecallResponse)
def recall_fragments(
    req: RecallRequest,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
    provider: EmbeddingProvider = Depends(get_provider),
) -> RecallResponse:
    """POST variant of search — supports full RecallRequest body."""
    return recall(req, storage, provider)


@router.get("/{fragment_id}", response_model=Fragment)
def get_fragment(
    fragment_id: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Fragment:
    frag = storage.get_fragment(fragment_id)
    if not frag:
        raise HTTPException(status_code=404, detail="Fragment not found")
    return frag


@router.patch("/{fragment_id}", response_model=Fragment)
def update_fragment(
    fragment_id: str,
    data: FragmentUpdate,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
    provider: EmbeddingProvider = Depends(get_provider),
) -> Fragment:
    """PATCH with OCC: include expected_version from your last GET."""
    frag = storage.get_fragment(fragment_id)
    if not frag:
        raise HTTPException(status_code=404, detail="Fragment not found")
    try:
        updated = storage.update_fragment(fragment_id, data)
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": str(e), "code": e.code},
        )
    # Re-embed if content changed
    if data.content:
        try:
            vec = provider.embed_one(updated.content)
            storage.set_fragment_embedding(fragment_id, vec_to_bytes(vec))
        except Exception:
            pass
    return updated


@router.delete("/{fragment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_fragment(
    fragment_id: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> None:
    """Soft-delete: marks as stale. Hard removal requires manual DB access."""
    if not storage.delete_fragment(fragment_id):
        raise HTTPException(status_code=404, detail="Fragment not found")
