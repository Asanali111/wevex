"""Clients pane — connected/available status for every supported LLM client."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual import work
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Static

from ..client import DaemonClient


class ClientsPane(Container):
    """Status table for claude_code, cursor, codex, gemini_cli, …"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Optional[DaemonClient] = None
        self._scope: str = ""

    def compose(self) -> ComposeResult:
        yield DataTable(id="clients-table", zebra_stripes=True, cursor_type="row")
        yield Static("", id="clients-status")

    def on_mount(self) -> None:
        table = self.query_one("#clients-table", DataTable)
        table.add_columns("client", "display name", "status")

    def bind_client(self, client: DaemonClient, scope: str) -> None:
        self._client = client
        self._scope = scope
        self.refresh_clients()

    def action_refresh(self) -> None:
        self.refresh_clients()

    @work(exclusive=True, group="clients-refresh")
    async def refresh_clients(self) -> None:
        if self._client is None:
            return
        status = self.query_one("#clients-status", Static)
        table = self.query_one("#clients-table", DataTable)
        try:
            entries: List[Dict[str, Any]] = await self._client.list_clients()
        except Exception as e:
            status.update(
                f"[red]Could not load client status: {type(e).__name__}: {e}[/red]"
            )
            return

        table.clear()
        connected = 0
        available = 0
        for d in entries:
            cid = d.get("id", "?")
            name = d.get("display_name", cid)
            is_connected = bool(d.get("connected"))
            is_detected = bool(d.get("detected"))
            if is_connected:
                state = "[green]● connected[/green]"
                connected += 1
            elif is_detected:
                state = "[yellow]○ available[/yellow]"
                available += 1
            else:
                state = "[dim]· not installed[/dim]"
            table.add_row(cid, name, state)
        status.update(
            f"[dim]{connected} connected · {available} available · "
            f"{len(entries)} total. Run [bold]skein connect[/bold] to wire one.[/dim]"
        )
