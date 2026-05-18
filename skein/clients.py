"""LLM client adapters — detect, connect, disconnect.

Each supported coding LLM has a ``BaseClient`` subclass that knows three
things:

  detect()      — is this client installed on this machine?
  connect()     — write the config files needed to point the client at Skein
  disconnect()  — surgically remove Skein's blocks from those config files

``ALL_CLIENTS`` is the registry — extending it by appending a new subclass is
the only step needed to support a new tool.

Why a separate module from ``sync.py``?
  ``sync.py`` was the original "blast everything to disk" routine. ``clients``
  splits the concerns: detection (so we can show the user what's installed),
  surgical disconnect (so we can cleanly remove individual clients), and per-
  client config logic (so adding the next tool is one class, not a sed pass).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("skein.clients")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _delete_if_empty_or_orphan(path: Path, key_chain: list[str]) -> bool:
    """If a JSON file's mcpServers (or other root) is empty after removing skein,
    leave the file but with empty containers — never delete user files."""
    return False  # placeholder; we never auto-delete


def _detect_path(*candidates: Path) -> tuple[bool, str]:
    """True if any candidate path exists."""
    for p in candidates:
        if p.exists():
            return True, f"found {p}"
    return False, "no install paths present"


def _detect_binary(*names: str) -> tuple[bool, str]:
    for n in names:
        if shutil.which(n):
            return True, f"binary {n!r} on PATH"
    return False, "no matching binary on PATH"


def _detect_any(*results: tuple[bool, str]) -> tuple[bool, str]:
    """OR-combine detect results."""
    found = [r for r in results if r[0]]
    if found:
        return True, found[0][1]
    return False, "; ".join(r[1] for r in results)


# ---------------------------------------------------------------------------
# BaseClient
# ---------------------------------------------------------------------------

class BaseClient:
    id: str = ""
    display_name: str = ""
    description: str = ""

    # Subclasses override
    def detect(self) -> tuple[bool, str]:
        raise NotImplementedError

    def connect(
        self,
        mcp_url: str,
        bearer_token: str,
        scope_handle: str,
        repo: Path,
    ) -> list[str]:
        """Write config(s); return list of paths written."""
        raise NotImplementedError

    def disconnect(self, recorded_paths: Optional[list[str]] = None) -> list[str]:
        """Remove skein from this client's config; return list of paths
        modified or cleaned. ``recorded_paths`` is the list captured at
        connect-time (from the connections registry); subclasses use it as
        a hint but should also defensively check default locations."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

class ClaudeCodeClient(BaseClient):
    id = "claude_code"
    display_name = "Claude Code"
    description = "Anthropic's official CLI"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("claude"),
            _detect_path(Path.home() / ".claude"),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        if not shutil.which("claude"):
            raise RuntimeError("claude binary not found in PATH")
        env = os.environ.copy()
        # If an old (header-less) entry exists, remove it first so the add
        # below doesn't no-op into the broken state. Idempotent.
        subprocess.run(
            ["claude", "mcp", "remove", "skein"],
            capture_output=True, text=True, timeout=15, env=env,
        )
        # Iter 16 fix: pass the bearer token via --header so Claude Code's
        # MCP client can authenticate against /mcp. Without this the entry
        # registers but every initialize returns 401 — exactly the "Failed
        # to connect" the user was seeing in `claude mcp list`.
        out = subprocess.run(
            [
                "claude", "mcp", "add", "skein", mcp_url,
                "--transport", "http",
                "--header", f"Authorization: Bearer {bearer_token}",
            ],
            capture_output=True, text=True, timeout=15, env=env,
        )
        if out.returncode != 0:
            txt = (out.stdout + out.stderr).lower()
            if "already" not in txt:
                raise RuntimeError(out.stderr.strip() or out.stdout.strip())
        # Claude stores MCPs in its own settings; we record a logical marker
        return [f"claude:mcp:skein@{mcp_url}"]

    def disconnect(self, recorded_paths=None) -> list[str]:
        if not shutil.which("claude"):
            return []
        try:
            subprocess.run(
                ["claude", "mcp", "remove", "skein"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception:
            pass
        return ["claude:mcp:skein"]


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

class CursorClient(BaseClient):
    id = "cursor"
    display_name = "Cursor"
    description = "AI-first IDE (Cursor.app)"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("cursor"),
            _detect_path(
                Path.home() / ".cursor",
                Path("/Applications/Cursor.app"),
            ),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        path = repo / ".cursor" / "mcp.json"
        data = _read_json(path)
        data.setdefault("mcpServers", {})
        data["mcpServers"]["skein"] = {
            "url": mcp_url,
            "type": "http",
            "headers": {"Authorization": f"Bearer {bearer_token}"},
        }
        _write_json(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        return _remove_skein_from_json(
            recorded_paths or [],
            ["mcpServers"],
            default_paths=[Path.cwd() / ".cursor" / "mcp.json"],
        )


# ---------------------------------------------------------------------------
# VS Code / GitHub Copilot
# ---------------------------------------------------------------------------

class VsCodeClient(BaseClient):
    id = "vscode"
    display_name = "VS Code / Copilot"
    description = "Visual Studio Code with GitHub Copilot Chat"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("code"),
            _detect_path(
                Path.home() / ".vscode",
                Path("/Applications/Visual Studio Code.app"),
            ),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        path = repo / ".vscode" / "mcp.json"
        data = _read_json(path)
        data.setdefault("mcpServers", {})
        data["mcpServers"]["skein"] = {
            "url": mcp_url,
            "type": "http",
            "headers": {"Authorization": f"Bearer {bearer_token}"},
        }
        _write_json(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        return _remove_skein_from_json(
            recorded_paths or [],
            ["mcpServers"],
            default_paths=[Path.cwd() / ".vscode" / "mcp.json"],
        )


# ---------------------------------------------------------------------------
# Gemini CLI
# ---------------------------------------------------------------------------

class GeminiCLIClient(BaseClient):
    id = "gemini_cli"
    display_name = "Gemini CLI"
    description = "Google's command-line Gemini agent"

    def detect(self) -> tuple[bool, str]:
        # Do not match Antigravity's nested ~/.gemini/antigravity dir alone.
        gemini_settings = Path.home() / ".gemini" / "settings.json"
        return _detect_any(
            _detect_binary("gemini"),
            _detect_path(gemini_settings),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        path = Path.home() / ".gemini" / "settings.json"
        data = _read_json(path)
        data.setdefault("mcpServers", {})
        # Gemini CLI's schema (settings.json) does NOT accept a "transport"
        # key under mcpServers — it infers HTTP vs stdio from the presence
        # of "url"/"httpUrl" vs "command". Including "transport" makes
        # Gemini CLI print a red "Unrecognized key(s)" warning at startup
        # even though the connection still works. Keep this object to the
        # documented keys only: url + headers.
        # See: https://geminicli.com/docs/reference/configuration/
        data["mcpServers"]["skein"] = {
            "url": mcp_url,
            "headers": {"Authorization": f"Bearer {bearer_token}"},
        }
        _write_json(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        return _remove_skein_from_json(
            recorded_paths or [],
            ["mcpServers"],
            default_paths=[Path.home() / ".gemini" / "settings.json"],
        )


# ---------------------------------------------------------------------------
# Antigravity
# ---------------------------------------------------------------------------

class AntigravityClient(BaseClient):
    id = "antigravity"
    display_name = "Antigravity"
    description = "Google's Antigravity (Electron-based agent IDE)"

    def detect(self) -> tuple[bool, str]:
        # Only the per-user config dir is a reliable signal — having
        # /Applications/Antigravity.app present without ever launching the
        # app would be a false positive.
        ag_dir = Path.home() / ".gemini" / "antigravity"
        return _detect_path(ag_dir)

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        ag_dir = Path.home() / ".gemini" / "antigravity"
        ag_dir.mkdir(parents=True, exist_ok=True)
        path = ag_dir / "mcp_config.json"
        # Defensive: if existing file is corrupt JSON, back it up before
        # overwriting so the user's prior content isn't silently lost.
        if path.exists():
            try:
                with open(path) as f:
                    json.load(f)
            except (json.JSONDecodeError, OSError):
                backup = path.with_suffix(".json.bak")
                path.rename(backup)
        data = _read_json(path)

        servers = data.setdefault("mcpServers", {})
        # Drop legacy company-brain entry from before the pivot
        servers.pop("company-brain", None)

        servers["skein"] = {
            "serverUrl": mcp_url,
            "headers": {"Authorization": f"Bearer {bearer_token}"},
        }
        _write_json(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        return _remove_skein_from_json(
            recorded_paths or [],
            ["mcpServers"],
            default_paths=[
                Path.home() / ".gemini" / "antigravity" / "mcp_config.json",
            ],
        )


# ---------------------------------------------------------------------------
# opencode
# ---------------------------------------------------------------------------

class OpenCodeClient(BaseClient):
    id = "opencode"
    display_name = "opencode"
    description = "Open-source TUI for AI coding agents"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("opencode"),
            _detect_path(Path.home() / ".config" / "opencode"),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        oc_dir = Path.home() / ".config" / "opencode"
        oc_dir.mkdir(parents=True, exist_ok=True)
        path = oc_dir / "config.json"
        data = _read_json(path)
        data.setdefault("mcp", {}).setdefault("servers", {})
        # opencode's MCP schema infers transport from the presence of `url` vs
        # `command` keys (same shape as the Gemini CLI fix in iter 18.1) — don't
        # write a separate `transport` field.
        data["mcp"]["servers"]["skein"] = {
            "url": mcp_url,
            "headers": {"Authorization": f"Bearer {bearer_token}"},
        }
        _write_json(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        return _remove_skein_from_json(
            recorded_paths or [],
            ["mcp", "servers"],
            default_paths=[
                Path.home() / ".config" / "opencode" / "config.json",
            ],
        )


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------

class CodexClient(BaseClient):
    id = "codex"
    display_name = "Codex CLI"
    description = "OpenAI Codex CLI / ChatGPT Desktop config"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("codex"),
            _detect_path(Path.home() / ".codex"),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        codex_dir = repo / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        path = codex_dir / "config.toml"
        existing = path.read_text() if path.exists() else ""

        # Iter 18.6+: strip any stale skein block before appending a fresh one,
        # so a token rotation propagates here instead of being silently ignored.
        # The previous "if skein in existing: return" guard meant the codex
        # config kept the dead leaked token after iter-16's rotation — caught
        # during iter 18 by a security sweep.
        cleaned = _strip_codex_skein_block(existing)

        # Codex's TOML schema infers transport from the presence of `url`
        # (same shape as the Gemini CLI fix in iter 18.1) — don't write a
        # separate `transport` field.
        block = (
            "\n[[mcpServers]]\n"
            'name = "skein"\n'
            f'url = "{mcp_url}"\n'
            "[mcpServers.headers]\n"
            f'Authorization = "Bearer {bearer_token}"\n'
        )
        # Ensure exactly one trailing newline before the new block, no double-
        # blank padding.
        body = cleaned.rstrip() + "\n" if cleaned.strip() else ""
        path.write_text(body + block)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        modified: list[str] = []
        candidates = [Path(p) for p in (recorded_paths or [])] or [
            Path.cwd() / ".codex" / "config.toml",
        ]
        for path in candidates:
            if not path.exists():
                continue
            text = path.read_text()
            if "skein" not in text:
                continue
            # Hand-rolled TOML edit — strip the [[mcpServers]] block whose
            # name = "skein" plus its [mcpServers.headers] sub-table.
            cleaned = _strip_codex_skein_block(text)
            if cleaned != text:
                path.write_text(cleaned)
                modified.append(str(path))
        return modified


def _strip_codex_skein_block(text: str) -> str:
    """Remove the skein-related TOML blocks from a codex config.

    The block we wrote on connect looks like::

        [[mcpServers]]
        name = "skein"
        url = "..."
        transport = "http"
        [mcpServers.headers]
        Authorization = "Bearer ..."

    We strip from ``[[mcpServers]]`` (with name = "skein") through the next
    blank line or top-level table header.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped == "[[mcpServers]]":
            # Look ahead to see if this block is the skein one.
            j = i + 1
            block: list[str] = [line]
            is_skein = False
            while j < n:
                nxt = lines[j]
                nxt_strip = nxt.strip()
                # End of this block: top-level table header that isn't our
                # nested headers, or a new [[mcpServers]], or EOF.
                if (
                    nxt_strip.startswith("[[")
                    or (
                        nxt_strip.startswith("[")
                        and not nxt_strip.startswith("[mcpServers.headers")
                    )
                ):
                    break
                if 'name = "skein"' in nxt_strip:
                    is_skein = True
                block.append(nxt)
                j += 1
            if is_skein:
                # Drop the block (and any single trailing blank line) entirely
                if j < n and lines[j].strip() == "":
                    j += 1
                i = j
                continue
            else:
                out.extend(block)
                i = j
                continue
        out.append(line)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Generic JSON disconnect helper
# ---------------------------------------------------------------------------

def _remove_skein_from_json(
    recorded_paths: list[str],
    key_chain: list[str],
    default_paths: list[Path],
) -> list[str]:
    """Walk into ``data[key_chain[0]][key_chain[1]]…`` and pop ``"skein"``.

    Tries ``recorded_paths`` first, falls back to ``default_paths``. Always
    leaves other entries intact. Empty parents are kept (so the user can
    inspect the file later)."""
    modified: list[str] = []
    candidates: list[Path] = []
    seen = set()
    for raw in (*recorded_paths, *(str(p) for p in default_paths)):
        p = Path(raw)
        if str(p) in seen:
            continue
        seen.add(str(p))
        candidates.append(p)

    for path in candidates:
        if not path.exists():
            continue
        data = _read_json(path)
        node = data
        for key in key_chain:
            if not isinstance(node, dict):
                break
            node = node.get(key, {})
        if isinstance(node, dict) and "skein" in node:
            del node["skein"]
            _write_json(path, data)
            modified.append(str(path))
    return modified


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_CLIENTS: list[BaseClient] = [
    ClaudeCodeClient(),
    CursorClient(),
    VsCodeClient(),
    GeminiCLIClient(),
    AntigravityClient(),
    OpenCodeClient(),
    CodexClient(),
]


def get_client(client_id: str) -> Optional[BaseClient]:
    for c in ALL_CLIENTS:
        if c.id == client_id:
            return c
    return None


def all_ids() -> list[str]:
    return [c.id for c in ALL_CLIENTS]


def detect_all() -> list[dict]:
    """Return ``[{id, display_name, description, detected, note}]`` for every
    known client."""
    out = []
    for c in ALL_CLIENTS:
        ok, note = c.detect()
        out.append({
            "id": c.id,
            "display_name": c.display_name,
            "description": c.description,
            "detected": ok,
            "note": note,
        })
    return out
