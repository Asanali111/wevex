"""Inbox pane — review pending extraction candidates."""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import DataTable, Static

from ..client import DaemonClient


class InboxPane(Container):
    """List of pending fragments + approve/reject chords."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("a", "approve", "approve", show=False),
        Binding("x", "reject", "reject", show=False),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Optional[DaemonClient] = None
        self._scope: str = ""
        self._candidates: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield DataTable(id="inbox-list", zebra_stripes=True, cursor_type="row")
        yield Static("[dim]Loading inbox…[/dim]", id="inbox-status")

    def on_mount(self) -> None:
        table = self.query_one("#inbox-list", DataTable)
        table.add_columns("id", "type", "conf", "tool", "preview")

    def bind_client(self, client: DaemonClient, scope: str) -> None:
        self._client = client
        self._scope = scope
        self.refresh_inbox()

    def action_refresh(self) -> None:
        self.refresh_inbox()

    def action_approve(self) -> None:
        cand = self._selected()
        if cand is None:
            return
        self._do_action(cand["id"], "approve")

    def action_reject(self) -> None:
        cand = self._selected()
        if cand is None:
            return
        self._do_action(cand["id"], "reject")

    def _selected(self) -> Optional[dict[str, Any]]:
        table = self.query_one("#inbox-list", DataTable)
        if table.cursor_row is None:
            return None
        idx = table.cursor_row
        if idx < 0 or idx >= len(self._candidates):
            return None
        return self._candidates[idx]

    @work(exclusive=True, group="inbox-refresh")
    async def refresh_inbox(self) -> None:
        if self._client is None:
            return
        status = self.query_one("#inbox-status", Static)
        table = self.query_one("#inbox-list", DataTable)
        try:
            candidates = await self._client.list_inbox(self._scope, limit=200)
        except Exception as e:
            status.update(f"[red]Could not load inbox: {type(e).__name__}: {e}[/red]")
            return

        self._candidates = list(candidates)
        table.clear()
        if not self._candidates:
            status.update(f"[dim]Inbox empty for[/dim] [bold]{self._scope}[/bold].")
            return
        for c in self._candidates:
            cid = (c.get("id") or "")[:8]
            ftype = c.get("type") or "—"
            conf = c.get("confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
            tool = c.get("source_tool") or "—"
            preview = (c.get("content") or "").splitlines()[0]
            if len(preview) > 80:
                preview = preview[:77] + "…"
            table.add_row(cid, ftype, conf_str, tool, preview)
        status.update(
            f"[dim]{len(self._candidates)} pending — [/dim]"
            "[bold #d97757]a[/bold #d97757][dim] approve · [/dim]"
            "[bold #d97757]x[/bold #d97757][dim] reject · [/dim]"
            "[bold #d97757]r[/bold #d97757][dim] refresh[/dim]"
        )

    @work(exclusive=True, group="inbox-act")
    async def _do_action(self, candidate_id: str, action: str) -> None:
        if self._client is None:
            return
        status = self.query_one("#inbox-status", Static)
        try:
            if action == "approve":
                result = await self._client.approve_candidate(candidate_id)
            else:
                result = await self._client.reject_candidate(candidate_id)
        except Exception as e:
            status.update(f"[red]Action failed: {type(e).__name__}: {e}[/red]")
            return
        rc = result.get("returncode", 0) if isinstance(result, dict) else 0
        if rc != 0:
            err = (result.get("stderr") or result.get("stdout") or "").strip()
            status.update(f"[red]{action} failed (rc={rc})[/red] [dim]{err}[/dim]")
            return
        status.update(f"[green]✓[/green] {action}d {candidate_id[:8]}…")
        # Re-pull the inbox so the candidate disappears.
        self.refresh_inbox()
