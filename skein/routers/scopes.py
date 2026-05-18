"""REST router: /v1/scopes"""
from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import AuthContext, RequireAuth
from ..dependencies import get_storage
from ..models import Scope, ScopeCreate, ScopeMembership, ScopeMembershipCreate
from ..storage import Storage

router = APIRouter(prefix="/v1/scopes", tags=["scopes"])


@router.post("", response_model=Scope, status_code=status.HTTP_201_CREATED)
def create_scope(
    data: ScopeCreate,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Scope:
    existing = storage.get_scope(data.handle)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Scope with handle '{data.handle}' already exists",
        )
    # Force owner to current user
    data.owner_id = auth.user_id
    return storage.create_scope(data)


@router.get("", response_model=list[Scope])
def list_scopes(
    limit: int = 50,
    offset: int = 0,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> list[Scope]:
    return storage.list_scopes(limit=limit, offset=offset)


@router.get("/{id_or_handle}", response_model=Scope)
def get_scope(
    id_or_handle: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Scope:
    scope = storage.get_scope(id_or_handle)
    if not scope:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scope not found")
    return scope


@router.get("/{id_or_handle}/lineage", response_model=list[Scope])
def get_scope_lineage(
    id_or_handle: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> list[Scope]:
    lineage = storage.get_scope_lineage(id_or_handle)
    if not lineage:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scope not found")
    return lineage


@router.post("/{scope_id}/members", response_model=ScopeMembership,
             status_code=status.HTTP_201_CREATED)
def add_member(
    scope_id: str,
    data: ScopeMembershipCreate,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> ScopeMembership:
    scope = storage.get_scope(scope_id)
    if not scope:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scope not found")
    data.scope_id = scope.id
    return storage.add_scope_member(data)
