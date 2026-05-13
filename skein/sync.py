"""Client config sync — orchestrator over ``skein.clients``.

``skein sync`` writes per-client MCP configs and, optionally, the universal
AGENTS.md / CLAUDE.md fallback. The actual per-client logic lives in
``skein.clients`` so the connect/disconnect surfaces stay symmetric.

By default, ``sync_all`` writes only for clients that the user has marked
connected (via ``skein connect`` → ``connections.json``). Pass
``client_ids=[...]`` to override (used internally by ``skein connect``
itself, before the registry is updated).
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import clients as clients_mod
from . import connections as conns

logger = logging.getLogger("skein.sync")


class SyncResult:
    """Tracks what was written and what was skipped."""

    def __init__(self) -> None:
        self.written: list[str] = []
        self.skipped: list[str] = []
        self.errors: list[str] = []

    def ok(self, label: str, path: str) -> None:
        self.written.append(f"{label}: {path}")

    def skip(self, label: str, reason: str) -> None:
        self.skipped.append(f"{label}: {reason}")

    def err(self, label: str, msg: str) -> None:
        self.errors.append(f"{label}: {msg}")


def sync_all(
    daemon_url: str,
    bearer_token: str,
    scope_handle: str,
    repo_path: Path | None = None,
    agents_md_content: str | None = None,
    client_ids: list[str] | None = None,
) -> SyncResult:
    """Write MCP configs for the requested clients and the universal fallback.

    Parameters
    ----------
    daemon_url:
        Base URL of the Skein daemon, e.g. "http://127.0.0.1:8765".
    bearer_token:
        Token for the daemon's Authorization: Bearer header.
    scope_handle:
        The project scope handle (written into AGENTS.md and some configs).
    repo_path:
        If provided, write AGENTS.md / CLAUDE.md / .cursor/mcp.json etc.
        relative to this directory.  Defaults to cwd.
    agents_md_content:
        Pre-rendered AGENTS.md content. If None, AGENTS.md is not written.
    client_ids:
        Explicit list of client IDs to sync. If None, syncs every client the
        user has previously connected via ``skein connect``.
    """
    mcp_url = f"{daemon_url}/mcp"
    repo = repo_path or Path.cwd()
    result = SyncResult()

    if client_ids is None:
        client_ids = conns.get_connected_ids()

    for cid in client_ids:
        client = clients_mod.get_client(cid)
        if client is None:
            result.skip(cid, "unknown client id")
            continue
        try:
            paths = client.connect(mcp_url, bearer_token, scope_handle, repo)
            for p in paths:
                result.ok(client.display_name, p)
            conns.mark_connected(cid, paths)
        except Exception as e:
            result.err(client.display_name, str(e))

    # ---- AGENTS.md (universal) ----
    if agents_md_content:
        _write_agents_md(repo, agents_md_content, result)

    # ---- CLAUDE.md shim (universal) ----
    _write_claude_md(repo, result)

    return result


def disconnect_client(client_id: str) -> list[str]:
    """Remove ``client_id`` from the registry and clean its config files.

    Returns the list of paths that were modified or removed.
    """
    client = clients_mod.get_client(client_id)
    if client is None:
        raise ValueError(f"Unknown client: {client_id}")

    recorded = conns.get_paths(client_id)
    modified = client.disconnect(recorded_paths=recorded)
    conns.mark_disconnected(client_id)
    return modified


# ---------------------------------------------------------------------------
# Universal fallback files (not tied to any single client)
# ---------------------------------------------------------------------------

def _write_agents_md(repo: Path, content: str, result: SyncResult) -> None:
    path = repo / "AGENTS.md"
    try:
        with open(path, "w") as f:
            f.write(content)
        result.ok("AGENTS.md", str(path))
    except Exception as e:
        result.err("AGENTS.md", str(e))


def _write_claude_md(repo: Path, result: SyncResult) -> None:
    path = repo / "CLAUDE.md"
    try:
        if path.exists():
            existing = path.read_text().strip()
            if existing not in ("@AGENTS.md", "@AGENTS.md\n"):
                result.skip("CLAUDE.md", "exists with non-shim content — not overwriting")
                return
        with open(path, "w") as f:
            f.write("@AGENTS.md\n")
        result.ok("CLAUDE.md", str(path))
    except Exception as e:
        result.err("CLAUDE.md", str(e))
