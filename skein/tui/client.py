"""Daemon client used by every TUI screen.

The TUI is async (Textual workers); the real implementation wraps
``httpx.AsyncClient``. Tests inject a mock implementing the same surface so
no screen needs a running daemon to render.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class DaemonClient(Protocol):
    """Async-callable surface every screen depends on.

    Methods raise ``httpx.ConnectError`` (or any ``Exception``) when the
    daemon is unreachable; screens catch broadly and show a banner.
    """

    async def health(self) -> Dict[str, Any]: ...
    async def briefing(self, scope: Optional[str]) -> Dict[str, Any]: ...
    async def recall(self, query: str, scope: str, limit: int = 10) -> Dict[str, Any]: ...
    async def list_clients(self) -> List[Dict[str, Any]]: ...
    async def list_inbox(self, scope: Optional[str], limit: int = 50) -> List[Dict[str, Any]]: ...
    async def approve_candidate(self, candidate_id: str) -> Dict[str, Any]: ...
    async def reject_candidate(self, candidate_id: str) -> Dict[str, Any]: ...
    async def read_events(self, limit: int = 100) -> List[Dict[str, Any]]: ...
    async def close(self) -> None: ...


class HttpDaemonClient:
    """Production client — talks to the FastAPI daemon over HTTP.

    Uses the bearer token + base URL from the same SkeinConfig the CLI reads.
    """

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        events_path: Optional[Path] = None,
    ) -> None:
        import httpx

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {bearer_token}"},
            timeout=10.0,
        )
        self._events_path = events_path

    async def health(self) -> Dict[str, Any]:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def briefing(self, scope: Optional[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if scope:
            params["scope"] = scope
        resp = await self._client.get("/v1/briefing", params=params)
        resp.raise_for_status()
        return resp.json()

    async def recall(self, query: str, scope: str, limit: int = 10) -> Dict[str, Any]:
        body = {"query": query, "scope": scope, "limit": limit}
        resp = await self._client.post("/v1/fragments/recall", json=body)
        resp.raise_for_status()
        return resp.json()

    async def list_clients(self) -> List[Dict[str, Any]]:
        from .. import clients as clients_mod
        from .. import connections as conns_mod

        detected = clients_mod.detect_all()
        connected = set(conns_mod.get_connected_ids())
        return [{**d, "connected": d["id"] in connected} for d in detected]

    async def list_inbox(self, scope: Optional[str], limit: int = 50) -> List[Dict[str, Any]]:
        from ..config import get_config
        from ..storage import Storage

        cfg = get_config()
        storage = Storage(cfg.db_path)
        try:
            scope_id: Optional[str] = None
            if scope:
                scope_obj = storage.get_scope(scope)
                if scope_obj:
                    scope_id = scope_obj.id
            return list(
                storage.list_extraction_candidates(scope_id=scope_id, limit=limit)
            )
        finally:
            storage.close()

    async def approve_candidate(self, candidate_id: str) -> Dict[str, Any]:
        return await self._candidate_action(candidate_id, "approve")

    async def reject_candidate(self, candidate_id: str) -> Dict[str, Any]:
        return await self._candidate_action(candidate_id, "reject")

    async def _candidate_action(self, candidate_id: str, action: str) -> Dict[str, Any]:
        import subprocess
        cmd = ["skein", "inbox", action, candidate_id]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    async def read_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        from ..events import default_path

        path = self._events_path or default_path()
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Factory + scope resolution
# ---------------------------------------------------------------------------

def build_default_client() -> HttpDaemonClient:
    """Build the production daemon client from ``SkeinConfig``."""
    from ..config import get_config

    cfg = get_config()
    return HttpDaemonClient(base_url=cfg.base_url, bearer_token=cfg.bearer_token)


def resolve_tui_scope(cli_scope: Optional[str]) -> str:
    """Resolve the active scope for the TUI without printing to stderr.

    Honors --scope > SKEIN_SCOPE > .skein/scope pin > config default.
    Mirrors ``cli._resolve_scope`` but never writes to stderr — that would
    garble the Textual screen.
    """
    from ..config import get_config
    from ..scope_resolver import resolve_scope

    cfg = get_config()
    scope, _source = resolve_scope(cli_scope, config_default=cfg.default_scope)
    return scope
