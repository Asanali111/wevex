"""The Skein TUI application root.

Single ``App`` instance with five tabs (Briefing, Fragments, Inbox, Events,
Clients), a persistent header (scope + health dot + counts), a footer with
chord shortcuts, and a help modal.

The app accepts an injected ``DaemonClient`` so tests can run without a live
daemon. In production we build the real ``HttpDaemonClient`` from
``SkeinConfig``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, ClassVar, List, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Static, TabbedContent, TabPane

from .client import DaemonClient, build_default_client, resolve_tui_scope
from .screens.briefing import BriefingPane
from .screens.clients import ClientsPane
from .screens.events import EventsPane
from .screens.fragments import FragmentsPane
from .screens.help_modal import HelpModal
from .screens.inbox import InboxPane
from .widgets.health_dot import HealthHeader

# Path to the bundled stylesheet — loaded via App.CSS_PATH.
_STYLES = Path(__file__).parent / "styles.css"


class SkeinApp(App):
    """Skein control-panel TUI.

    Parameters
    ----------
    scope:
        Optional explicit scope handle. Falls back to standard scope
        resolution (env > pin > config default).
    client_factory:
        Callable returning a ``DaemonClient``. Default factory builds the
        real HTTP client from config. Tests inject one that returns canned
        data or raises connection errors.
    """

    CSS_PATH = str(_STYLES)
    TITLE = "Skein"
    SUB_TITLE = "context bus"

    BINDINGS: ClassVar[List[Binding]] = [
        Binding("q", "quit", "quit", show=True),
        Binding("?", "help", "help", show=True),
        Binding("r", "refresh", "refresh", show=True),
        Binding("1", "go_tab('briefing')", "briefing", show=False),
        Binding("2", "go_tab('fragments')", "fragments", show=False),
        Binding("3", "go_tab('inbox')", "inbox", show=False),
        Binding("4", "go_tab('events')", "events", show=False),
        Binding("5", "go_tab('clients')", "clients", show=False),
        Binding("/", "focus_search", "search", show=False),
    ]

    def __init__(
        self,
        scope: Optional[str] = None,
        client_factory: Optional[Callable[[], DaemonClient]] = None,
    ) -> None:
        super().__init__()
        # Defer scope resolution: in test mode we don't want config side effects.
        self._explicit_scope = scope
        self._client_factory = client_factory
        self._client: Optional[DaemonClient] = None
        self._scope: str = scope or "personal:scratch"

    # ------------------------------------------------------------------ compose

    def compose(self) -> ComposeResult:
        yield HealthHeader(id="health-header")
        yield Static("", id="daemon-banner", classes="-hidden")
        with TabbedContent(initial="briefing", id="tabs"):
            with TabPane("Briefing", id="briefing"):
                yield BriefingPane(id="briefing-pane")
            with TabPane("Fragments", id="fragments"):
                yield FragmentsPane(id="fragments-pane")
            with TabPane("Inbox", id="inbox"):
                yield InboxPane(id="inbox-pane")
            with TabPane("Events", id="events"):
                yield EventsPane(id="events-pane")
            with TabPane("Clients", id="clients"):
                yield ClientsPane(id="clients-pane")
        yield Footer()

    # ------------------------------------------------------------------- mount

    async def on_mount(self) -> None:
        # Build client + resolve scope only now — keeps __init__ side-effect free
        # for tests that pass everything in.
        if self._client is None:
            self._client = self._make_client()
        if self._explicit_scope is None:
            try:
                self._scope = resolve_tui_scope(None)
            except Exception:
                self._scope = "personal:scratch"

        header = self.query_one(HealthHeader)
        header.scope = self._scope

        # Hand each pane the client + scope.
        self.query_one(BriefingPane).bind_client(self._client, self._scope)
        self.query_one(FragmentsPane).bind_client(self._client, self._scope)
        self.query_one(InboxPane).bind_client(self._client, self._scope)
        self.query_one(EventsPane).bind_client(self._client, self._scope)
        self.query_one(ClientsPane).bind_client(self._client, self._scope)

        # Kick off the first health refresh; periodic interval follows.
        self.refresh_health()
        self.set_interval(5.0, self.refresh_health)

    # ----------------------------------------------------------------- helpers

    def _make_client(self) -> DaemonClient:
        if self._client_factory is not None:
            return self._client_factory()
        return build_default_client()

    @work(exclusive=True, group="health")
    async def refresh_health(self) -> None:
        """Refresh header + show banner if daemon is unreachable."""
        header = self.query_one(HealthHeader)
        banner = self.query_one("#daemon-banner", Static)
        assert self._client is not None
        try:
            briefing = await self._client.briefing(self._scope)
            header.healthy = True
            header.chunks_total = briefing.get("chunks_total")
            header.fragment_total = briefing.get("fragment_total")
            header.embedding_provider = briefing.get("embedding_provider")
            banner.update("")
            banner.add_class("-hidden")
        except Exception:
            header.healthy = False
            banner.update(
                "Skein daemon not running. Start it with [bold]skein up[/bold] "
                "in your project directory."
            )
            banner.remove_class("-hidden")

    # ------------------------------------------------------------------ actions

    def action_quit(self) -> None:
        self.exit()

    def action_help(self) -> None:
        self.push_screen(HelpModal())

    def action_refresh(self) -> None:
        # Re-mount-style refresh of whichever pane is showing.
        self.refresh_health()
        tabs = self.query_one(TabbedContent)
        active = tabs.active
        try:
            pane = self.query_one(f"#{active}-pane")
        except Exception:
            return
        refresh = getattr(pane, "action_refresh", None)
        if callable(refresh):
            refresh()

    def action_go_tab(self, tab_id: str) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = tab_id
        except Exception:
            pass

    def action_focus_search(self) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            if tabs.active == "fragments":
                pane = self.query_one(FragmentsPane)
                pane.focus_search()
        except Exception:
            pass

    # ----------------------------------------------------------------- cleanup

    async def on_unmount(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
