"""REST router: /v1/identities"""
from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import AuthContext, RequireAuth
from ..dependencies import get_storage
from ..models import Identity, IdentityCreate
from ..storage import Storage

router = APIRouter(prefix="/v1/identities", tags=["identities"])


@router.post("", response_model=Identity, status_code=status.HTTP_201_CREATED)
def create_identity(
    data: IdentityCreate,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Identity:
    existing = storage.get_identity(data.handle)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Identity with handle '{data.handle}' already exists",
        )
    return storage.create_identity(data)


@router.get("", response_model=list[Identity])
def list_identities(
    limit: int = 50,
    offset: int = 0,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> list[Identity]:
    return storage.list_identities(limit=limit, offset=offset)


@router.get("/{id_or_handle}", response_model=Identity)
def get_identity(
    id_or_handle: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Identity:
    identity = storage.get_identity(id_or_handle)
    if not identity:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Identity not found")
    return identity
