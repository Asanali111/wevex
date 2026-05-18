"""Fragments pane — recall / hybrid search across the scope."""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Static

from ..client import DaemonClient


class FragmentsPane(Container):
    """Recall input + result table; row click opens detail modal."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Optional[DaemonClient] = None
        self._scope: str = ""
        self._results: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="fragments-search-bar"):
            yield Input(
                placeholder="Search Skein fragments… (Enter to recall)",
                id="fragments-search-input",
            )
        yield Static("[dim]Type a query and press Enter.[/dim]", id="fragments-status")
        yield DataTable(id="fragments-results", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#fragments-results", DataTable)
        table.add_columns("#", "score", "type", "tool", "preview")

    def bind_client(self, client: DaemonClient, scope: str) -> None:
        self._client = client
        self._scope = scope

    def focus_search(self) -> None:
        try:
            self.query_one("#fragments-search-input", Input).focus()
        except Exception:
            pass

    def action_refresh(self) -> None:
        # No-op — refresh on Fragments tab re-runs the current query.
        try:
            inp = self.query_one("#fragments-search-input", Input)
        except Exception:
            return
        if inp.value.strip():
            self.run_query(inp.value.strip())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "fragments-search-input":
            return
        q = (event.value or "").strip()
        if not q:
            return
        self.run_query(q)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "fragments-results":
            return
        idx = event.cursor_row
        if idx is None or idx < 0 or idx >= len(self._results):
            return
        frag = self._results[idx]
        self.app.push_screen(FragmentDetailModal(frag))

    @work(exclusive=True, group="fragments-recall")
    async def run_query(self, query: str) -> None:
        if self._client is None:
            return
        status = self.query_one("#fragments-status", Static)
        table = self.query_one("#fragments-results", DataTable)
        status.update(f"[dim]Searching for[/dim] [bold]{query}[/bold]…")
        try:
            payload = await self._client.recall(query, self._scope, limit=20)
        except Exception as e:
            status.update(f"[red]Recall failed: {type(e).__name__}: {e}[/red]")
            return

        hits = payload.get("hits") or payload.get("results") or []
        self._results = list(hits)
        table.clear()
        if not hits:
            status.update(
                f"[dim]No hits for[/dim] [bold]{query}[/bold] "
                f"[dim]in scope[/dim] [bold]{self._scope}[/bold]."
            )
            return
        for rank, hit in enumerate(hits, start=1):
            score = hit.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            ftype = hit.get("type") or "—"
            tool = hit.get("created_by_tool") or "—"
            content = hit.get("content") or ""
            preview = content.splitlines()[0] if content else ""
            if len(preview) > 80:
                preview = preview[:77] + "…"
            table.add_row(str(rank), score_str, ftype, tool, preview)
        status.update(
            f"[dim]Returned[/dim] [bold]{len(hits)}[/bold] "
            f"[dim]hits for[/dim] [bold]{query}[/bold]."
        )


class FragmentDetailModal(ModalScreen):
    """Read-only popup with the full fragment content and metadata."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss", "close"),
    ]

    def __init__(self, fragment: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fragment = fragment

    def compose(self) -> ComposeResult:
        f = self._fragment
        meta_bits = []
        for k in ("id", "type", "scope", "created_by_tool", "created_at"):
            if k in f:
                meta_bits.append(f"[bold]{k}[/bold]: {f[k]}")
        meta = "  ·  ".join(meta_bits)
        content = f.get("content", "")
        body = (
            f"{meta}\n\n"
            f"{content}\n\n"
            "[dim](escape to close)[/dim]"
        )
        with Container(id="fragment-detail-dialog"):
            yield Static(body)

    def action_dismiss(self) -> None:
        self.app.pop_screen()
