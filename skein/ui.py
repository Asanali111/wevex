"""Shared terminal UI primitives.

The visual language is deliberately spare:

  ●  / ○         — filled / empty status dot
  ✓  / ✗         — done / failed
  ─              — soft horizontal rule
  ·              — bullet
  cyan           — identifiers (scope handles, client ids)
  green / red    — success / failure
  yellow         — warning, in-progress
  dim            — secondary metadata

Layouts use whitespace, not borders. Avoid Panel/box for primary output —
keep them for the very final "ready" summary only.

All helpers print to a shared ``rich.console.Console``. Pass ``console=...``
to override (used in tests and for `--json` paths that route to stderr).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

from rich.console import Console

# Two consoles: stdout for normal output, stderr for warnings/errors.
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Status dot — the single most-reused element
# ---------------------------------------------------------------------------

_DOT = {
    "ok":   ("●", "green"),
    "err":  ("●", "red"),
    "warn": ("●", "yellow"),
    "info": ("●", "cyan"),
    "idle": ("○", "dim"),
    "off":  ("○", "red"),
}


def dot(state: str = "ok") -> str:
    """Return a styled status dot as a rich markup string."""
    glyph, color = _DOT.get(state, _DOT["ok"])
    return f"[{color}]{glyph}[/{color}]"


# ---------------------------------------------------------------------------
# Status mark — green ✓ / red ✗ / yellow → / dim ·
# ---------------------------------------------------------------------------

_MARK = {
    "ok":   ("✓", "green"),
    "err":  ("✗", "red"),
    "warn": ("⚠", "yellow"),
    "step": ("→", "yellow"),
    "skip": ("·", "dim"),
}


def mark(state: str = "ok") -> str:
    glyph, color = _MARK.get(state, _MARK["ok"])
    return f"[{color}]{glyph}[/{color}]"


# ---------------------------------------------------------------------------
# Section header — bold title with a status dot
# ---------------------------------------------------------------------------

def header(title: str, *, state: str = "ok", subtitle: Optional[str] = None) -> None:
    console.print()
    line = f"  {dot(state)} [bold]{title}[/bold]"
    if subtitle:
        line += f"  [dim]{subtitle}[/dim]"
    console.print(line)
    console.print()


def section(title: str) -> None:
    """Lightweight bold heading — no dot."""
    console.print()
    console.print(f"  [bold]{title}[/bold]")


def divider() -> None:
    console.print(f"  [dim]{'─' * 56}[/dim]")


# ---------------------------------------------------------------------------
# Aligned field rows
# ---------------------------------------------------------------------------

def field(label: str, value: str, *, label_width: int = 12, indent: int = 2) -> None:
    """Print one ``  Label        value`` row."""
    pad = " " * indent
    console.print(f"{pad}[dim]{label:<{label_width}}[/dim]  {value}")


def fields(
    pairs: Sequence[Tuple[str, str]],
    *,
    label_width: Optional[int] = None,
    indent: int = 2,
) -> None:
    """Print a block of label/value rows; auto-sizes the label column."""
    if not pairs:
        return
    if label_width is None:
        label_width = max(len(p[0]) for p in pairs)
    for label, value in pairs:
        field(label, value, label_width=label_width, indent=indent)


# ---------------------------------------------------------------------------
# Bullets and step rows
# ---------------------------------------------------------------------------

def bullet(text: str, *, indent: int = 4, mark_str: str = "·",
           mark_color: str = "dim") -> None:
    pad = " " * indent
    console.print(f"{pad}[{mark_color}]{mark_str}[/{mark_color}] {text}")


def step(text: str, *, state: str = "ok", detail: Optional[str] = None,
         indent: int = 2) -> None:
    """Print ``  ✓ text   detail`` — one line per finished step."""
    pad = " " * indent
    line = f"{pad}{mark(state)} {text}"
    if detail:
        line += f"  [dim]{detail}[/dim]"
    console.print(line)


# ---------------------------------------------------------------------------
# Two-column status list (e.g. `skein clients`)
# ---------------------------------------------------------------------------

def status_list(
    rows: Iterable[Tuple[str, str, str, str]],
    *,
    indent: int = 2,
) -> None:
    """Render rows of ``(dot_state, id, name, note)`` as an aligned three-column
    list. Auto-widths every column.

    Example output:
        ●  claude_code  Claude Code        connected
        ○  cursor       Cursor             not installed
    """
    rows = list(rows)
    if not rows:
        return
    id_w = max(len(r[1]) for r in rows)
    name_w = max(len(r[2]) for r in rows)
    pad = " " * indent
    for state, ident, name, note in rows:
        console.print(
            f"{pad}{dot(state)}  "
            f"[cyan]{ident:<{id_w}}[/cyan]  "
            f"{name:<{name_w}}  "
            f"[dim]{note}[/dim]"
        )


# ---------------------------------------------------------------------------
# Final summary — the one place we still use a panel
# ---------------------------------------------------------------------------

def panel_ready(title: str, body: str) -> None:
    """Final 'ready' card. Used by `skein up` and similar terminal moments."""
    from rich.panel import Panel
    from rich.box import ROUNDED
    console.print()
    console.print(Panel(
        body,
        title=f"[green]{title}[/green]",
        title_align="left",
        border_style="dim",
        box=ROUNDED,
        padding=(0, 2),
    ))


# ---------------------------------------------------------------------------
# Footer / hint line
# ---------------------------------------------------------------------------

def hint(text: str) -> None:
    console.print()
    console.print(f"  [dim]{text}[/dim]")


def blank() -> None:
    console.print()


def home_relative(path: str) -> str:
    """Replace the leading $HOME with ``~`` for display."""
    from pathlib import Path
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


# ---------------------------------------------------------------------------
# Status counter line — dim summary footer
# ---------------------------------------------------------------------------

def counter_line(parts: Sequence[Tuple[int, str]], *, indent: int = 2) -> None:
    """Print ``  3 detected · 2 connected · 4 not installed``.

    Each (n, label) becomes ``n label``. Empty-counter parts are dropped."""
    chunks = [f"{n} {label}" for n, label in parts if n]
    if not chunks:
        return
    pad = " " * indent
    console.print(f"{pad}[dim]{' · '.join(chunks)}[/dim]")
