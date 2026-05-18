"""Events pane — live tail of ``~/.config/skein/events.jsonl``.

Polls the JSONL directly (no HTTP); each tick reads the trailing N events
and renders them top-down newest-first. ``p`` toggles polling, ``c`` clears
the visible buffer.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.timer import Timer
from textual.widgets import DataTable, Static

from ..client import DaemonClient

_POLL_SECONDS = 2.0


class EventsPane(Container):
    """Live event tail."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("p", "toggle_pause", "pause", show=False),
        Binding("c", "clear", "clear", show=False),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Optional[DaemonClient] = None
        self._scope: str = ""
        self._paused: bool = False
        self._poll_timer: Optional[Timer] = None
        # Track which event lines we've already shown (by (ts, event, scope)).
        self._seen: set[tuple[str, str, str]] = set()

    def compose(self) -> ComposeResult:
        yield DataTable(id="events-log", zebra_stripes=True, cursor_type="row")
        yield Static(
            "[dim]Tailing events… [bold]p[/bold] to pause, "
            "[bold]c[/bold] to clear.[/dim]",
            id="events-status",
        )

    def on_mount(self) -> None:
        table = self.query_one("#events-log", DataTable)
        table.add_columns("ts", "event", "scope", "detail")

    def bind_client(self, client: DaemonClient, scope: str) -> None:
        self._client = client
        self._scope = scope
        # Initial read + start polling.
        self.poll_events()
        self._poll_timer = self.set_interval(_POLL_SECONDS, self._tick)

    def action_refresh(self) -> None:
        self.poll_events()

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        status = self.query_one("#events-status", Static)
        if self._paused:
            status.update(
                "[bold #d97757]Paused.[/bold #d97757] [dim]Press p to resume.[/dim]"
            )
        else:
            status.update(
                "[dim]Tailing events… [bold]p[/bold] to pause, "
                "[bold]c[/bold] to clear.[/dim]"
            )

    def action_clear(self) -> None:
        table = self.query_one("#events-log", DataTable)
        table.clear()
        self._seen.clear()

    def _tick(self) -> None:
        if self._paused:
            return
        self.poll_events()

    @work(exclusive=True, group="events-poll")
    async def poll_events(self) -> None:
        if self._client is None:
            return
        try:
            records: list[dict[str, Any]] = await self._client.read_events(limit=100)
        except Exception as e:
            status = self.query_one("#events-status", Static)
            status.update(f"[red]Could not read events: {type(e).__name__}: {e}[/red]")
            return

        table = self.query_one("#events-log", DataTable)
        new_rows = 0
        for rec in records:
            ts = str(rec.get("ts", "?"))
            ev = str(rec.get("event", "?"))
            sc = str(rec.get("scope") or "—")
            key = (ts, ev, sc)
            if key in self._seen:
                continue
            self._seen.add(key)
            details = rec.get("details") or {}
            detail_str = _summarise_details(details)
            table.add_row(ts, ev, sc, detail_str)
            new_rows += 1
        if new_rows:
            # Scroll to the bottom so the newest row is visible.
            table.action_scroll_end()


def _summarise_details(details: dict[str, Any]) -> str:
    bits = []
    for k in ("query", "preview", "glob"):
        v = details.get(k)
        if v:
            if isinstance(v, str) and len(v) > 30:
                v = v[:27] + "…"
            bits.append(f"{k}={v!r}")
    for k in ("hits", "type", "fragment_id", "lease_id"):
        if k in details:
            v = details[k]
            if isinstance(v, str) and len(v) > 10:
                v = v[:8] + "…"
            bits.append(f"{k}={v}")
    return " ".join(bits)
