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
import sys
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("skein.clients")


def _is_windows() -> bool:
    return sys.platform.startswith("win") or os.name == "nt"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _appdata_dir(name: str) -> Optional[Path]:
    """Return ``%APPDATA%\\name`` on Windows, else None.

    Used by clients (opencode, …) that follow XDG on POSIX but live under
    Roaming AppData on Windows.
    """
    if not _is_windows():
        return None
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / name


def _localappdata_dir(name: str) -> Optional[Path]:
    """Return ``%LOCALAPPDATA%\\name`` on Windows, else None.

    Most Windows GUI apps install themselves under LocalAppData (Cursor,
    VS Code's user install). Detection-only — not used as a config target.
    """
    if not _is_windows():
        return None
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / name
    return Path.home() / "AppData" / "Local" / name


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


def _write_hermes_env_key(env_path: Path, key: str, value: str) -> None:
    """Write or update KEY=VALUE in a ~/.hermes/.env file."""
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={value}\n")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("".join(lines), encoding="utf-8")


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
        candidates = [Path.home() / ".cursor"]
        # macOS — Cursor.app bundle in /Applications. Always-present
        # on POSIX, must not be probed on Windows where Path("/Applications/…")
        # resolves to "C:\Applications\…" and silently always-misses.
        if _is_macos():
            candidates.append(Path("/Applications/Cursor.app"))
        # Windows — Cursor's user install lands in %LOCALAPPDATA%\Programs\cursor.
        win_install = _localappdata_dir("Programs") and _localappdata_dir("Programs") / "cursor"
        if win_install is not None:
            candidates.append(win_install)
        return _detect_any(
            _detect_binary("cursor"),
            _detect_path(*candidates),
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
        candidates = [Path.home() / ".vscode"]
        if _is_macos():
            candidates.append(Path("/Applications/Visual Studio Code.app"))
        # Windows — system install at %ProgramFiles%\Microsoft VS Code,
        # user install at %LOCALAPPDATA%\Programs\Microsoft VS Code.
        if _is_windows():
            for base_env in ("ProgramFiles", "ProgramFiles(x86)"):
                base = os.environ.get(base_env)
                if base:
                    candidates.append(Path(base) / "Microsoft VS Code")
            user_install = _localappdata_dir("Programs")
            if user_install is not None:
                candidates.append(user_install / "Microsoft VS Code")
        return _detect_any(
            _detect_binary("code"),
            _detect_path(*candidates),
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

def _opencode_config_dir() -> Path:
    """opencode's config dir is XDG on POSIX, %APPDATA%\\opencode on Windows.

    Mirrors the upstream opencode behaviour — writing to ``~/.config/opencode``
    on Windows would land in a folder opencode doesn't read, so the daemon
    would still work but Skein wouldn't actually be wired up.
    """
    win = _appdata_dir("opencode")
    if win is not None:
        return win
    return Path.home() / ".config" / "opencode"


class OpenCodeClient(BaseClient):
    id = "opencode"
    display_name = "opencode"
    description = "Open-source TUI for AI coding agents"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("opencode"),
            _detect_path(_opencode_config_dir()),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        oc_dir = _opencode_config_dir()
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
            default_paths=[_opencode_config_dir() / "config.json"],
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
# Goose (by Block)
# ---------------------------------------------------------------------------

def _goose_config_dir() -> Path:
    """Return Goose's config directory.

    - macOS / Linux: ``~/.config/goose/`` (etcetera XDG strategy, app_name only)
    - Windows: ``%APPDATA%\\Block\\goose\\`` (etcetera Windows strategy)

    Source: ``crates/goose/src/config/paths.rs`` + ``config-files.md`` in the
    block/goose repository (verified May 2026).
    """
    if _is_windows():
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Block" / "goose"
    return Path.home() / ".config" / "goose"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError):
        return {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


class GooseClient(BaseClient):
    id = "goose"
    display_name = "Goose"
    description = "Block's open-source local-first AI agent"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("goose"),
            _detect_path(_goose_config_dir()),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        cfg_dir = _goose_config_dir()
        cfg_dir.mkdir(parents=True, exist_ok=True)
        path = cfg_dir / "config.yaml"
        data = _read_yaml(path)
        data.setdefault("extensions", {})
        # Goose's streamable_http extension schema (ExtensionConfig in
        # crates/goose/src/agents/extension.rs, serde rename = "streamable_http"):
        #   enabled, type, name, description, uri, headers, timeout
        data["extensions"]["skein"] = {
            "enabled": True,
            "type": "streamable_http",
            "name": "skein",
            "description": "Skein MCP context bus",
            "uri": mcp_url,
            "headers": {"Authorization": f"Bearer {bearer_token}"},
            "timeout": 300,
        }
        _write_yaml(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        modified: list[str] = []
        candidates: list[Path] = []
        seen: set[str] = set()
        default = _goose_config_dir() / "config.yaml"
        for raw in (*( recorded_paths or []), str(default)):
            if raw in seen:
                continue
            seen.add(raw)
            candidates.append(Path(raw))

        for path in candidates:
            if not path.exists():
                continue
            data = _read_yaml(path)
            exts = data.get("extensions", {})
            if isinstance(exts, dict) and "skein" in exts:
                del exts["skein"]
                _write_yaml(path, data)
                modified.append(str(path))
        return modified


# ---------------------------------------------------------------------------
# gptme
# ---------------------------------------------------------------------------

def _gptme_config_path() -> Path:
    """gptme uses ``~/.config/gptme/config.toml`` on all platforms.

    Unlike opencode/Cursor which have Windows-specific AppData paths, gptme
    hardcodes ``os.path.expanduser("~/.config/gptme/config.toml")`` in its
    source (gptme/config/user.py) — no platform branching.
    """
    return Path.home() / ".config" / "gptme" / "config.toml"


def _strip_gptme_skein_block(text: str) -> str:
    """Remove the skein ``[[mcp.servers]]`` block from a gptme config.

    The block we write on connect looks like::

        [[mcp.servers]]
        name = "skein"
        enabled = true
        url = "..."
        headers = { Authorization = "Bearer ..." }

    We strip from the ``[[mcp.servers]]`` line whose content includes
    ``name = "skein"`` through the next blank line or top-level table header.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped == "[[mcp.servers]]":
            # Peek ahead to see if this block belongs to skein
            j = i + 1
            block: list[str] = [line]
            is_skein = False
            while j < n:
                nxt = lines[j]
                nxt_strip = nxt.strip()
                # End of this block: any top-level table header or new array-of-tables
                if nxt_strip.startswith("[[") or (
                    nxt_strip.startswith("[") and not nxt_strip.startswith("[[")
                ):
                    break
                if 'name = "skein"' in nxt_strip:
                    is_skein = True
                block.append(nxt)
                j += 1
            if is_skein:
                # Drop the block and any single trailing blank line
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


class GptmeClient(BaseClient):
    id = "gptme"
    display_name = "gptme"
    description = "Autonomous terminal agent"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("gptme"),
            _detect_path(_gptme_config_path().parent),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        path = _gptme_config_path()
        existing = path.read_text() if path.exists() else ""

        # Strip any stale skein block before appending a fresh one so that a
        # token rotation propagates instead of being silently ignored
        # (same lesson as the iter-18.6 Codex fix).
        cleaned = _strip_gptme_skein_block(existing)

        # gptme TOML schema (docs/mcp.rst): [[mcp.servers]] with name, enabled,
        # url, and inline-table headers. Transport inferred from presence of url.
        block = (
            "\n[[mcp.servers]]\n"
            'name = "skein"\n'
            "enabled = true\n"
            f'url = "{mcp_url}"\n'
            f'headers = {{Authorization = "Bearer {bearer_token}"}}\n'
        )
        body = cleaned.rstrip() + "\n" if cleaned.strip() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + block)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        modified: list[str] = []
        candidates = [Path(p) for p in (recorded_paths or [])] or [
            _gptme_config_path(),
        ]
        for path in candidates:
            if not path.exists():
                continue
            text = path.read_text()
            if "skein" not in text:
                continue
            cleaned = _strip_gptme_skein_block(text)
            if cleaned != text:
                path.write_text(cleaned)
                modified.append(str(path))
        return modified


# ---------------------------------------------------------------------------
# Windsurf
# ---------------------------------------------------------------------------

class WindsurfClient(BaseClient):
    id = "windsurf"
    display_name = "Windsurf"
    description = "Codeium's AI-native IDE"

    def detect(self) -> tuple[bool, str]:
        candidates = [Path.home() / ".codeium" / "windsurf"]
        if _is_macos():
            candidates.append(Path("/Applications/Windsurf.app"))
        return _detect_any(
            _detect_binary("windsurf"),
            _detect_path(*candidates),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        path = repo / ".windsurf" / "mcp.json"
        data = _read_json(path)
        data.setdefault("mcpServers", {})
        data["mcpServers"]["skein"] = {
            "serverUrl": mcp_url,   # Windsurf uses "serverUrl" not "url"
            "headers": {"Authorization": f"Bearer {bearer_token}"},
        }
        _write_json(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        return _remove_skein_from_json(
            recorded_paths or [],
            ["mcpServers"],
            default_paths=[Path.cwd() / ".windsurf" / "mcp.json"],
        )


# ---------------------------------------------------------------------------
# Hermes (Nous Research)
# ---------------------------------------------------------------------------

class HermesClient(BaseClient):
    id = "hermes"
    display_name = "Hermes"
    description = "Nous Research's autonomous AI agent"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("hermes"),
            _detect_path(Path.home() / ".hermes"),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        import yaml
        hermes_home = Path.home() / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        config_path = hermes_home / "config.yaml"
        env_path = hermes_home / ".env"

        # Write token to .env
        _write_hermes_env_key(env_path, "MCP_SKEIN_API_KEY", bearer_token)

        # Update config.yaml
        config = {}
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                config = {}
        config.setdefault("mcp_servers", {})["skein"] = {
            "url": mcp_url,
            "headers": {"Authorization": "Bearer ${MCP_SKEIN_API_KEY}"},
        }
        tmp = config_path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.dump(config, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(tmp, config_path)
        return [str(config_path), str(env_path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        import yaml
        config_path = Path.home() / ".hermes" / "config.yaml"
        env_path = Path.home() / ".hermes" / ".env"
        modified = []
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                config = {}
            servers = config.get("mcp_servers", {})
            if isinstance(servers, dict) and "skein" in servers:
                del servers["skein"]
                if not servers:
                    config.pop("mcp_servers", None)
                tmp = config_path.with_suffix(".yaml.tmp")
                tmp.write_text(
                    yaml.dump(config, default_flow_style=False, allow_unicode=True),
                    encoding="utf-8",
                )
                os.replace(tmp, config_path)
                modified.append(str(config_path))
        if env_path.exists():
            _write_hermes_env_key(env_path, "MCP_SKEIN_API_KEY", "")
            modified.append(str(env_path))
        return modified


# ---------------------------------------------------------------------------
# Crush (Charm)
# ---------------------------------------------------------------------------

class CrushClient(BaseClient):
    id = "crush"
    display_name = "Crush"
    description = "Charm's terminal coding agent"

    def detect(self) -> tuple[bool, str]:
        return _detect_any(
            _detect_binary("crush"),
            _detect_path(Path.home() / ".config" / "crush"),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        # Crush resolves .crush.json (project-local hidden) first in its
        # priority order: .crush.json > crush.json > $XDG_CONFIG_HOME/crush/crush.json.
        # Writing to the project-local hidden file keeps user's global config intact.
        path = repo / ".crush.json"
        data = _read_json(path)
        data.setdefault("mcp", {})
        # Crush requires "type" to be stated explicitly — it does NOT infer
        # transport from key presence (unlike Gemini CLI / opencode).
        data["mcp"]["skein"] = {
            "type": "http",
            "url": mcp_url,
            "headers": {"Authorization": f"Bearer {bearer_token}"},
        }
        _write_json(path, data)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        return _remove_skein_from_json(
            recorded_paths or [],
            ["mcp"],
            default_paths=[Path.cwd() / ".crush.json"],
        )


# ---------------------------------------------------------------------------
# Kiro
# ---------------------------------------------------------------------------

class KiroClient(BaseClient):
    id = "kiro"
    display_name = "Kiro"
    description = "AWS's spec-first AI IDE"

    def detect(self) -> tuple[bool, str]:
        candidates = [Path.home() / ".kiro"]
        if _is_macos():
            candidates.append(Path("/Applications/Kiro.app"))
        return _detect_any(
            _detect_binary("kiro"),
            _detect_path(*candidates),
        )

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        # Kiro workspace config lives at .kiro/settings/mcp.json (note the
        # extra settings/ segment — different from Cursor's .cursor/mcp.json).
        # Kiro's schema infers transport from the presence of "url" vs
        # "command" — no explicit "type" field, per kiro.dev/docs/mcp/configuration/.
        path = repo / ".kiro" / "settings" / "mcp.json"
        data = _read_json(path)
        data.setdefault("mcpServers", {})
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
            default_paths=[Path.cwd() / ".kiro" / "settings" / "mcp.json"],
        )


# ---------------------------------------------------------------------------
# Continue.dev
# ---------------------------------------------------------------------------

class ContinueClient(BaseClient):
    id = "continue"
    display_name = "Continue.dev"
    description = "Open-source AI code assistant for VS Code / JetBrains"

    # Continue.dev picks up standalone block files from ~/.continue/mcpServers/.
    # Documented at docs.continue.dev/customize/deep-dives/mcp (Quick Start).
    # Using a dedicated file avoids touching the user's hand-edited config.yaml
    # and makes disconnect = delete one file.
    _BLOCK_FILENAME = "skein.yaml"

    def _mcpservers_dir(self) -> Path:
        return Path.home() / ".continue" / "mcpServers"

    def _block_path(self) -> Path:
        return self._mcpservers_dir() / self._BLOCK_FILENAME

    def detect(self) -> tuple[bool, str]:
        # ~/.continue is uniquely owned by the Continue.dev VS Code/JetBrains
        # extension — it's a reliable detection signal.
        return _detect_path(Path.home() / ".continue")

    def connect(self, mcp_url, bearer_token, scope_handle, repo) -> list[str]:
        import yaml  # pyyaml — already a project dep

        block_dir = self._mcpservers_dir()
        block_dir.mkdir(parents=True, exist_ok=True)
        path = self._block_path()

        # Overwrite unconditionally — same lesson as iter 18.6 Codex fix
        # (stale tokens must not survive a token rotation).
        data = {
            "name": "Skein",
            "version": "0.0.1",
            "schema": "v1",
            "mcpServers": [
                {
                    "name": "skein",
                    "type": "streamable-http",
                    "url": mcp_url,
                    "requestOptions": {
                        "headers": {
                            "Authorization": f"Bearer {bearer_token}",
                        },
                    },
                }
            ],
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return [str(path)]

    def disconnect(self, recorded_paths=None) -> list[str]:
        # The block file is entirely Skein-owned — safe to delete outright.
        candidates: list[Path] = []
        seen: set[str] = set()
        for raw in (recorded_paths or []):
            p = Path(raw)
            key = str(p)
            if key not in seen:
                seen.add(key)
                candidates.append(p)
        # Always try the default location as a fallback.
        default = self._block_path()
        if str(default) not in seen:
            candidates.append(default)

        removed: list[str] = []
        for path in candidates:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        return removed


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
    GooseClient(),
    GptmeClient(),
    WindsurfClient(),
    HermesClient(),
    CrushClient(),
    KiroClient(),
    ContinueClient(),
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
