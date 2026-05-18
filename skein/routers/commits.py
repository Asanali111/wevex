"""REST router: /v1/commits — read-only (append-only log)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import AuthContext, RequireAuth
from ..dependencies import get_storage
from ..models import Commit
from ..storage import Storage

router = APIRouter(prefix="/v1/commits", tags=["commits"])


@router.get("", response_model=list[Commit])
def list_commits(
    scope: Optional[str] = Query(None, description="Scope handle or ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = 0,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> list[Commit]:
    scope_id: Optional[str] = None
    if scope:
        scope_obj = storage.get_scope(scope)
        if not scope_obj:
            raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
        scope_id = scope_obj.id
    return storage.list_commits(scope_id=scope_id, limit=limit, offset=offset)


@router.get("/{commit_id}", response_model=Commit)
def get_commit(
    commit_id: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Commit:
    commit = storage.get_commit(commit_id)
    if not commit:
        raise HTTPException(status_code=404, detail="Commit not found")
    return commit
