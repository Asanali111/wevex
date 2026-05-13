"""Help modal — shows chord shortcuts grouped by screen."""
from __future__ import annotations

from typing import ClassVar, List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

_HELP_BODY = """[bold]Global[/bold]
  [bold cyan]q[/bold cyan]       quit
  [bold cyan]?[/bold cyan]       this help
  [bold cyan]r[/bold cyan]       refresh the active tab
  [bold cyan]1-5[/bold cyan]     jump to tab (briefing · fragments · inbox · events · clients)
  [bold cyan]/[/bold cyan]       focus the search input (Fragments only)

[bold]Fragments[/bold]
  Type, then [bold cyan]Enter[/bold cyan]    run a recall
  [bold cyan]Enter[/bold cyan]               open the highlighted hit
  [bold cyan]escape[/bold cyan]              close detail popup

[bold]Inbox[/bold]
  [bold cyan]a[/bold cyan]       approve highlighted candidate
  [bold cyan]x[/bold cyan]       reject highlighted candidate

[bold]Events[/bold]
  [bold cyan]p[/bold cyan]       pause / resume polling
  [bold cyan]c[/bold cyan]       clear the visible buffer
"""

_FOOTER = (
    "Skein v0.1.0  ·  MIT  ·  https://github.com/Asanali111/skein"
)


class HelpModal(ModalScreen):
    """Floating shortcut reference; dismissed with escape, ?, or q."""

    BINDINGS: ClassVar[List[Binding]] = [
        Binding("escape", "dismiss", "close"),
        Binding("?", "dismiss", "close"),
        Binding("q", "dismiss", "close"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="help-dialog"):
            yield Static("Skein TUI — Help", id="help-title")
            yield Static(_HELP_BODY, id="help-body")
            yield Static(_FOOTER, id="help-footer")

    def action_dismiss(self) -> None:
        self.app.pop_screen()
