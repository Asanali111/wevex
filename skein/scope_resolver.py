"""Single source of truth for resolving the active scope handle.

Resolution order (first match wins):
  1. Explicit ``cli_scope`` argument (e.g. from a ``--scope`` flag)
  2. ``SKEIN_SCOPE`` env var (set by hooks_install for Claude Code hooks)
  3. ``.skein/scope`` file in cwd or any parent (written by ``skein hooks install``)
  4. ``cfg.default_scope``

This is shared between the hook handlers (skein/hooks.py) and the human-facing
CLI commands (skein/cli.py) so a project-level pin applies uniformly to every
``skein …`` invocation, not just the ones called from inside Claude Code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


def auto_detect_scope(start: Optional[Path] = None) -> str:
    """Best-guess scope handle for a project directory.

    Priority:
      1. Existing ``.skein/scope`` file in cwd or any parent.
      2. ``project:<git-remote-basename>`` if the current dir is a git repo.
      3. ``project:<cwd-basename>`` as a final fallback.

    Refuses to invent a project handle for the user's $HOME, ``/``, or
    ``/tmp`` — those are not projects, and auto-creating ``project:<homename>``
    on every Claude Code session-start in ``~`` was the cause of the
    long-standing junk-scope leak. Falls back to ``personal:scratch`` instead.
    """
    import re
    import subprocess

    pin = find_scope_pin(start)
    if pin:
        return pin

    cwd = (start or Path.cwd()).resolve()

    # Refuse to invent a project handle for non-project dirs.
    if _is_non_project_dir(cwd):
        return "personal:scratch"

    # Try git remote
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), "remote", "get-url", "origin"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        # Examples:
        #   git@github.com:user/repo.git          → repo
        #   https://github.com/user/repo.git      → repo
        #   /Users/me/code/repo                   → repo
        m = re.search(r"[/:]([A-Za-z0-9_.\-]+?)(?:\.git)?/?$", out)
        if m and m.group(1):
            return f"project:{_clean_handle_part(m.group(1))}"
    except Exception:
        pass

    return f"project:{_clean_handle_part(cwd.name)}"


def _is_non_project_dir(cwd: Path) -> bool:
    """Heuristic: this directory is not a real project root."""
    cwd = cwd.resolve()
    home = Path.home().resolve()
    if cwd == home:
        return True
    if str(cwd) in {"/", "/tmp", "/var", "/private/tmp"}:
        return True
    # iCloud / Dropbox / Documents top-level — also not project roots
    if cwd in {home / "Documents", home / "Desktop", home / "Downloads"}:
        return True
    return False


def _clean_handle_part(s: str) -> str:
    """Lowercase + strip characters that don't make for nice handles."""
    import re
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", s.lower()).strip("-")
    return cleaned or "default"


def find_scope_pin(start: Optional[Path] = None) -> Optional[str]:
    """Walk up from ``start`` (or cwd) looking for a ``.skein/scope`` file.

    Returns the pinned scope handle, or None if no pin file is found.
    """
    cwd = (start or Path.cwd()).resolve()
    for parent in [cwd, *cwd.parents]:
        f = parent / ".skein" / "scope"
        if f.exists():
            try:
                value = f.read_text().strip()
                if value:
                    return value
            except OSError:
                continue
    return None


def resolve_scope(
    cli_scope: Optional[str] = None,
    *,
    config_default: Optional[str] = None,
    start: Optional[Path] = None,
) -> Tuple[str, str]:
    """Resolve the scope to use, plus a one-word source label for telemetry.

    Returns ``(scope_handle, source)`` where ``source`` is one of:
      - ``"cli"``  — explicit --scope flag
      - ``"env"``  — SKEIN_SCOPE env var
      - ``"pin"``  — .skein/scope file
      - ``"config"`` — cfg.default_scope fallback

    Raises ``RuntimeError`` if no scope can be determined.
    """
    if cli_scope:
        return cli_scope, "cli"

    env_scope = os.environ.get("SKEIN_SCOPE")
    if env_scope:
        return env_scope, "env"

    pin = find_scope_pin(start)
    if pin:
        return pin, "pin"

    if config_default:
        return config_default, "config"

    raise RuntimeError(
        "Could not resolve scope. Pass --scope, set SKEIN_SCOPE, "
        "create a .skein/scope file (run `skein hooks install`), "
        "or set default_scope in your config."
    )
