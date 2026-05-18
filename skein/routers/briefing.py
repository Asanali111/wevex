"""REST router: /v1/briefing — single-call project state snapshot.

Mirror of the ``project_briefing`` MCP tool. Same payload, exposed over HTTP
so the CLI (``skein briefing``) and any non-MCP consumer can hit it directly.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query

from ..auth import AuthContext, RequireAuth
from ..dependencies import get_storage
from ..mcp import build_briefing
from ..scope_resolver import resolve_scope
from ..storage import Storage

router = APIRouter(prefix="/v1/briefing", tags=["briefing"])


@router.get("", response_model=dict[str, Any])
def get_briefing(
    scope: Optional[str] = Query(
        None, description="Scope handle, e.g. 'project:myapp'. Omit to auto-detect."
    ),
    auth: AuthContext = RequireAuth,
    storage: Storage = Depends(get_storage),
) -> dict[str, Any]:
    """Return the project's current state in one round trip.

    See :func:`skein.mcp.build_briefing` for the response shape.
    """
    from ..config import get_config

    if not scope:
        cfg = get_config()
        try:
            scope, _ = resolve_scope(None, config_default=cfg.default_scope)
        except Exception:
            scope = cfg.default_scope or "personal:scratch"
    return build_briefing(storage, scope)
