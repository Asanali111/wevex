"""Skein adapter against a live, already-running daemon.

Read-only — never mutates the user's real data. Reads the bearer token and
port from ``~/.config/skein/config.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import httpx

from ..adapter import (
    CodeChunkResult,
    FragmentResult,
    HealthInfo,
    ReadOnlyAdapter,
)


class SkeinLiveAdapter(ReadOnlyAdapter):
    """Hits the live Skein daemon at ``http://host:port`` over real HTTP."""

    name = "skein-live"
    supports_typed_fragments = True
    supports_leases = True
    supports_code_search = True
    supports_scope_hierarchy = True
    supports_git_capture = True

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        cfg = self._load_config()
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 8765)
        self.base_url = base_url or f"http://{host}:{port}"
        self.token = token or cfg.get("bearer_token", "")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
            timeout=httpx.Timeout(60.0),
        )

    @staticmethod
    def _load_config() -> dict:
        path = Path.home() / ".config" / "skein" / "config.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def close(self) -> None:
        self._client.close()

    # ---- ReadOnlyAdapter ------------------------------------------------

    def health(self) -> HealthInfo:
        r = self._client.get("/health")
        r.raise_for_status()
        d = r.json()
        return HealthInfo(
            fragment_count=d.get("fragment_count", 0),
            chunk_count=d.get("chunk_count", 0),
            scope_count=d.get("scope_count", 0),
            version=d.get("version", ""),
            tool=self.name,
            extra=d,
        )

    def recall(
        self,
        query: str,
        scope: str,
        *,
        limit: int = 10,
        types: Optional[List[str]] = None,
    ) -> List[FragmentResult]:
        body = {"query": query, "scope": scope, "limit": limit}
        if types:
            body["types"] = types
        r = self._client.post("/v1/fragments/recall", json=body)
        r.raise_for_status()
        out = []
        for item in r.json().get("results", []):
            frag = item.get("fragment", {})
            out.append(FragmentResult(
                id=frag.get("id", ""),
                content=frag.get("content", ""),
                type=frag.get("type", ""),
                score=float(item.get("score", 0.0)),
            ))
        return out

    def search_code(
        self,
        query: str,
        scope: str,
        *,
        limit: int = 10,
    ) -> List[CodeChunkResult]:
        body = {"query": query, "scope": scope, "limit": limit}
        r = self._client.post("/v1/chunks/search", json=body)
        r.raise_for_status()
        out = []
        for item in r.json().get("results", []):
            chunk = item.get("chunk", {})
            out.append(CodeChunkResult(
                id=chunk.get("id", ""),
                content=chunk.get("content", ""),
                file_path=chunk.get("file_path", ""),
                score=float(item.get("score", 0.0)),
            ))
        return out
