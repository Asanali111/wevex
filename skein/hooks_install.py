"""Install/uninstall autonomous hooks for Claude Code, Cursor, and friends.

What this writes (when target tools are detected):

  Claude Code (per-project):
    .claude/settings.json        — merged with existing; adds Skein hooks
    .skein/scope                 — pins the project scope handle for hooks

  Claude Code (user-global, optional):
    ~/.claude/settings.json      — same merge, applies to all projects

  Cursor (per-project):
    .cursor/rules/skein.mdc      — auto-applied rule pointing at skein

  Codex CLI / Gemini CLI / opencode / Antigravity:
    rely on AGENTS.md (already written by `skein sync`); no extra hook file.

The merge is conservative: existing skein entries are replaced; user keys
outside the skein-owned blocks are preserved.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("skein.hooks_install")

# Marker so we can find and remove our own entries idempotently
_SKEIN_MARKER_KEY = "__skein_managed"


@dataclass
class InstallReport:
    written: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def ok(self, label: str, path: str) -> None:
        self.written.append(f"{label}: {path}")

    def skip(self, label: str, reason: str) -> None:
        self.skipped.append(f"{label}: {reason}")

    def err(self, label: str, msg: str) -> None:
        self.errors.append(f"{label}: {msg}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_hooks(
    repo_path: Path,
    scope_handle: str,
    *,
    skein_bin: str = "skein",
    user_global: bool = False,
) -> InstallReport:
    """Install autonomous hooks for all detected clients."""
    report = InstallReport()

    # 1. Pin the scope for this project (read by hook handlers)
    _write_scope_pin(repo_path, scope_handle, report)

    # 2. Claude Code project hooks
    _install_claude_code(repo_path, scope_handle, skein_bin, report)
    if user_global:
        _install_claude_code_global(scope_handle, skein_bin, report)

    # 3. Cursor rule
    _install_cursor_rule(repo_path, scope_handle, skein_bin, report)

    return report


def uninstall_hooks(repo_path: Path) -> InstallReport:
    """Remove Skein-managed hooks (preserves user-added entries)."""
    report = InstallReport()

    # Remove .skein/scope (and the dir if empty)
    scope_pin = repo_path / ".skein" / "scope"
    if scope_pin.exists():
        scope_pin.unlink()
        try:
            scope_pin.parent.rmdir()
        except OSError:
            pass
        report.ok(".skein/scope", str(scope_pin))

    # Strip from .claude/settings.json
    claude_settings = repo_path / ".claude" / "settings.json"
    if claude_settings.exists():
        _strip_skein_from_claude_settings(claude_settings, report)

    # Remove .cursor/rules/skein.mdc
    cursor_rule = repo_path / ".cursor" / "rules" / "skein.mdc"
    if cursor_rule.exists():
        cursor_rule.unlink()
        report.ok("Cursor rule", str(cursor_rule))

    return report


# ---------------------------------------------------------------------------
# Scope pin
# ---------------------------------------------------------------------------

def _write_scope_pin(repo_path: Path, scope_handle: str, report: InstallReport) -> None:
    try:
        skein_dir = repo_path / ".skein"
        skein_dir.mkdir(exist_ok=True)
        (skein_dir / "scope").write_text(scope_handle + "\n")
        report.ok("Scope pin", str(skein_dir / "scope"))
    except Exception as e:
        report.err("Scope pin", str(e))


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

def _install_claude_code(
    repo_path: Path, scope_handle: str, skein_bin: str, report: InstallReport,
) -> None:
    settings_path = repo_path / ".claude" / "settings.json"
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings = _read_json_or_empty(settings_path)
        _merge_claude_skein_hooks(settings, skein_bin, scope_handle)
        _write_json(settings_path, settings)
        report.ok("Claude Code (project)", str(settings_path))
    except Exception as e:
        report.err("Claude Code (project)", str(e))


def _install_claude_code_global(
    scope_handle: str, skein_bin: str, report: InstallReport,
) -> None:
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings = _read_json_or_empty(settings_path)
        # User-global doesn't pin scope; hooks rely on per-project .skein/scope
        _merge_claude_skein_hooks(settings, skein_bin, scope_handle=None)
        _write_json(settings_path, settings)
        report.ok("Claude Code (global)", str(settings_path))
    except Exception as e:
        report.err("Claude Code (global)", str(e))


def _merge_claude_skein_hooks(
    settings: dict, skein_bin: str, scope_handle: Optional[str],
) -> None:
    """Merge Skein hook entries into a Claude Code settings dict, idempotently.

    Format follows the Claude Code 'hooks' schema:
      { "hooks": { "<EventName>": [ {"matcher": "*", "hooks": [{"type":"command", "command":"..."}]} ] } }
    """
    hooks_root = settings.setdefault("hooks", {})
    env_prefix = f"SKEIN_SCOPE={scope_handle} " if scope_handle else ""

    events = {
        "SessionStart":     f"{env_prefix}{skein_bin} hook session-start",
        "UserPromptSubmit": f"{env_prefix}{skein_bin} hook user-prompt-submit",
        "Stop":             f"{env_prefix}{skein_bin} hook stop",
        "PostToolUse":      f"{env_prefix}{skein_bin} hook post-tool-use",
    }

    for event_name, command in events.items():
        existing_blocks = hooks_root.setdefault(event_name, [])
        # Remove any prior Skein-managed block for this event
        existing_blocks[:] = [
            b for b in existing_blocks
            if not (isinstance(b, dict) and b.get(_SKEIN_MARKER_KEY))
        ]
        # Add ours
        block = {
            _SKEIN_MARKER_KEY: True,
            "matcher": "*",
            "hooks": [{"type": "command", "command": command}],
        }
        existing_blocks.append(block)


def _strip_skein_from_claude_settings(path: Path, report: InstallReport) -> None:
    try:
        settings = _read_json_or_empty(path)
        hooks_root = settings.get("hooks", {})
        changed = False
        for event_name, blocks in list(hooks_root.items()):
            if not isinstance(blocks, list):
                continue
            new_blocks = [
                b for b in blocks
                if not (isinstance(b, dict) and b.get(_SKEIN_MARKER_KEY))
            ]
            if len(new_blocks) != len(blocks):
                changed = True
                if new_blocks:
                    hooks_root[event_name] = new_blocks
                else:
                    del hooks_root[event_name]
        if changed:
            if not hooks_root:
                settings.pop("hooks", None)
            _write_json(path, settings)
            report.ok("Claude Code (cleaned)", str(path))
        else:
            report.skip("Claude Code", "no Skein-managed hooks found")
    except Exception as e:
        report.err("Claude Code", str(e))


# ---------------------------------------------------------------------------
# Cursor rule (.cursor/rules/skein.mdc)
# ---------------------------------------------------------------------------

_CURSOR_RULE_TEMPLATE = """---
description: Skein context bus integration — call recall/remember automatically
alwaysApply: true
---

# Skein integration

This project uses **Skein** for cross-LLM context sharing. The local daemon
exposes an MCP server you have access to.

## Use these tools proactively

- **At the start of any non-trivial task**, call the `recall` MCP tool with a
  query that summarises the task. Treat the returned fragments as authoritative
  context that other agents have left for you.
- **After each significant decision**, call `remember` (or `note_decision`) to
  persist it. Use `type="decision"` for choices, `"observation"` for code-level
  changes, `"requirement"` for hard rules, `"preference"` for style.
- **Before editing files in a shared area**, call `claim_lease` with the file
  glob, so other agents won't clobber your work.

## Scope

Use scope handle `{scope}` for this project.

## Fallback

If the MCP server is unavailable, run shell commands instead:
```
{skein_bin} recall "<query>"
{skein_bin} remember "<content>" --type decision
```

(This file is auto-managed by `skein hooks install`. Delete it or run
`skein hooks uninstall` to remove.)
"""


def _install_cursor_rule(
    repo_path: Path, scope_handle: str, skein_bin: str, report: InstallReport,
) -> None:
    rules_dir = repo_path / ".cursor" / "rules"
    try:
        rules_dir.mkdir(parents=True, exist_ok=True)
        rule_path = rules_dir / "skein.mdc"
        content = _CURSOR_RULE_TEMPLATE.format(scope=scope_handle, skein_bin=skein_bin)
        rule_path.write_text(content)
        report.ok("Cursor rule", str(rule_path))
    except Exception as e:
        report.err("Cursor rule", str(e))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _read_json_or_empty(path: Path) -> dict:
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
