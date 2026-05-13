"""Persistent header status indicator.

Renders ``Skein · <scope> · ◉ daemon: <status> · <chunks> chunks · <frags> frags``.
Lives across all screens; refreshed by ``SkeinApp``'s periodic worker.
"""
from __future__ import annotations

from typing import Optional

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class HealthHeader(Static):
    """Single-line header showing scope + daemon health + summary counts."""

    healthy: reactive[bool] = reactive(False)
    scope: reactive[str] = reactive("personal:scratch")
    chunks_total: reactive[Optional[int]] = reactive(None)
    fragment_total: reactive[Optional[int]] = reactive(None)
    embedding_provider: reactive[Optional[str]] = reactive(None)

    DEFAULT_CSS = """
    HealthHeader {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
    }
    """

    def render(self) -> Text:
        dot_color = "#7ec27e" if self.healthy else "#d97757"
        status = "healthy" if self.healthy else "down"
        text = Text()
        text.append("Skein", style="bold #d97757")
        text.append("  ·  ", style="dim")
        text.append(self.scope, style="cyan")
        text.append("  ·  ", style="dim")
        text.append("◉", style=f"bold {dot_color}")
        text.append(f" daemon: {status}", style="bold")
        if self.chunks_total is not None:
            text.append("  ·  ", style="dim")
            text.append(f"{self.chunks_total} chunks", style="dim")
        if self.fragment_total is not None:
            text.append("  ·  ", style="dim")
            text.append(f"{self.fragment_total} frags", style="dim")
        if self.embedding_provider:
            text.append("  ·  ", style="dim")
            text.append(self.embedding_provider, style="italic dim")
        return text
