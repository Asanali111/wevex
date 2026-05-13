"""Briefing pane — the default landing tab.

Renders the same payload as ``skein briefing`` but live in a tabbed surface:
fragment counts by type, recent decisions, daemon stats, recommended action.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from textual import work
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Static

from ..client import DaemonClient


class BriefingPane(Container):
    """Briefing screen body."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Optional[DaemonClient] = None
        self._scope: str = ""

    def compose(self) -> ComposeResult:
        yield Static("[bold]Loading briefing…[/bold]", id="briefing-header")
        yield Static("", id="briefing-stats")
        yield DataTable(id="briefing-counts", zebra_stripes=True, cursor_type="row")
        yield Static("", id="briefing-decisions")
        yield Static("", id="briefing-action")

    def on_mount(self) -> None:
        table = self.query_one("#briefing-counts", DataTable)
        table.add_columns("type", "count")

    def bind_client(self, client: DaemonClient, scope: str) -> None:
        self._client = client
        self._scope = scope
        self.refresh_briefing()

    def action_refresh(self) -> None:
        self.refresh_briefing()

    @work(exclusive=True, group="briefing")
    async def refresh_briefing(self) -> None:
        if self._client is None:
            return
        header = self.query_one("#briefing-header", Static)
        stats = self.query_one("#briefing-stats", Static)
        table = self.query_one("#briefing-counts", DataTable)
        decisions = self.query_one("#briefing-decisions", Static)
        action = self.query_one("#briefing-action", Static)

        try:
            briefing: Dict[str, Any] = await self._client.briefing(self._scope)
        except Exception as e:
            header.update(f"[bold]Skein — {self._scope}[/bold]")
            stats.update(
                "[red]Could not load briefing.[/red]\n"
                f"[dim]{type(e).__name__}: {e}[/dim]"
            )
            return

        header.update(f"[bold]Skein — {briefing.get('scope', self._scope)}[/bold]")

        daemon = briefing.get("daemon", {})
        uptime = daemon.get("uptime_seconds")
        uptime_str = _fmt_uptime(uptime) if isinstance(uptime, (int, float)) else "—"
        stats_lines = [
            f"[bold]fragment_total[/bold]: {briefing.get('fragment_total', 0)}",
            f"[bold]chunks_total[/bold]: {briefing.get('chunks_total', 0)}",
            f"[bold]active_inbox_count[/bold]: {briefing.get('active_inbox_count', 0)}",
            f"[bold]embedding_provider[/bold]: {briefing.get('embedding_provider', '—')}",
            f"[bold]daemon uptime[/bold]: {uptime_str}",
            f"[bold]daemon version[/bold]: {daemon.get('version', '—')}",
        ]
        stats.update("\n".join(stats_lines))

        table.clear()
        counts = briefing.get("fragment_counts", {}) or {}
        for ftype, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            table.add_row(ftype, str(count))

        recent = briefing.get("recent_decisions", []) or []
        if recent:
            lines = ["[bold]Recent decisions[/bold]"]
            for d in recent:
                short = d.get("id_short", "—")
                line = d.get("content_first_line", "")
                tool = d.get("created_by_tool", "—")
                lines.append(
                    f"  [yellow]{short}[/yellow]  [dim]{tool:>14}[/dim]  {line}"
                )
            decisions.update("\n".join(lines))
        else:
            decisions.update(
                "[bold]Recent decisions[/bold]\n  [dim]none yet — "
                "use [bold]skein note[/bold] or call the [bold]note_decision[/bold] "
                "MCP tool to record one.[/dim]"
            )

        next_action = briefing.get("next_recommended_action") or ""
        if next_action:
            action.update(f"[bold #d97757]Next:[/bold #d97757] {next_action}")
        else:
            action.update("")


def _fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
