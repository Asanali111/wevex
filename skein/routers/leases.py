"""REST router: /v1/leases — advisory file-glob locks."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import AuthContext, RequireAuth
from ..dependencies import get_storage
from ..models import Lease, LeaseCreate
from ..storage import Storage

router = APIRouter(prefix="/v1/leases", tags=["leases"])


@router.post("", response_model=Lease, status_code=status.HTTP_201_CREATED)
def acquire_lease(
    data: LeaseCreate,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Lease:
    """Acquire an advisory lease on a file-glob pattern.

    Returns 409 if a conflicting lease is already active.
    """
    # Validate scope
    scope = storage.get_scope(data.scope_id)
    if not scope:
        raise HTTPException(status_code=404, detail=f"Scope '{data.scope_id}' not found")
    data.scope_id = scope.id
    data.owner_id = auth.user_id

    conflict = storage.check_lease_conflict(scope.id, data.glob)
    if conflict and conflict.owner_id != auth.user_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "Lease conflict",
                "code": "LEASE_CONFLICT",
                "conflicting_lease_id": conflict.id,
                "held_by": conflict.owner_id,
                "glob": conflict.glob,
                "expires_at": conflict.expires_at,
            },
        )

    return storage.acquire_lease(data)


@router.get("", response_model=list[Lease])
def list_leases(
    scope: Optional[str] = Query(None, description="Scope handle or ID"),
    active_only: bool = True,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> list[Lease]:
    scope_id: Optional[str] = None
    if scope:
        scope_obj = storage.get_scope(scope)
        if not scope_obj:
            raise HTTPException(status_code=404, detail=f"Scope '{scope}' not found")
        scope_id = scope_obj.id
    return storage.list_leases(scope_id=scope_id, active_only=active_only)


@router.get("/{lease_id}", response_model=Lease)
def get_lease(
    lease_id: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> Lease:
    lease = storage.get_lease(lease_id)
    if not lease:
        raise HTTPException(status_code=404, detail="Lease not found")
    return lease


@router.delete("/{lease_id}", status_code=status.HTTP_204_NO_CONTENT)
def release_lease(
    lease_id: str,
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> None:
    """Release a lease.  Only the lease owner can release it."""
    if not storage.release_lease(lease_id, auth.user_id):
        # Either not found or not owner
        lease = storage.get_lease(lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Lease not found")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not the owner of this lease",
        )
