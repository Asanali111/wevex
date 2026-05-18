"""Help modal — shows chord shortcuts grouped by screen."""
from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

_HELP_BODY = """[bold]Global[/bold]
  [bold #d97757]q[/bold #d97757]       quit
  [bold #d97757]?[/bold #d97757]       this help
  [bold #d97757]r[/bold #d97757]       refresh the active tab
  [bold #d97757]1-5[/bold #d97757]     jump to tab (briefing · fragments · inbox · events · clients)
  [bold #d97757]/[/bold #d97757]       focus the search input (Fragments only)

[bold]Fragments[/bold]
  Type, then [bold #d97757]Enter[/bold #d97757]    run a recall
  [bold #d97757]Enter[/bold #d97757]               open the highlighted hit
  [bold #d97757]escape[/bold #d97757]              close detail popup

[bold]Inbox[/bold]
  [bold #d97757]a[/bold #d97757]       approve highlighted candidate
  [bold #d97757]x[/bold #d97757]       reject highlighted candidate

[bold]Events[/bold]
  [bold #d97757]p[/bold #d97757]       pause / resume polling
  [bold #d97757]c[/bold #d97757]       clear the visible buffer
"""

_FOOTER = (
    "Skein v0.1.0  ·  MIT  ·  https://github.com/Asanali111/skein"
)


class HelpModal(ModalScreen):
    """Floating shortcut reference; dismissed with escape, ?, or q."""

    BINDINGS: ClassVar[list[Binding]] = [
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
