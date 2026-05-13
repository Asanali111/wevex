"""Tests for the Textual TUI (iter 21).

The TUI is async and screen-based, so every test instantiates ``SkeinApp``
with an injected mock ``DaemonClient`` (real daemon never required). The
Textual ``Pilot`` harness drives keypresses and inspects the live widget
tree.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import pytest

from skein.tui.app import SkeinApp

# ---------------------------------------------------------------------------
# Mock daemon client
# ---------------------------------------------------------------------------

_CANNED_BRIEFING: Dict[str, Any] = {
    "scope": "project:test-scope",
    "fragment_counts": {
        "decision": 3, "fact": 2, "observation": 1, "preference": 0,
        "state": 0, "requirement": 0, "procedure": 0, "conversation": 0,
    },
    "fragment_total": 6,
    "chunks_total": 42,
    "recent_decisions": [
        {
            "id_short": "abc12345",
            "content_first_line": "use async/await for all I/O",
            "created_by_tool": "claude_code",
            "created_at": "2026-05-13T10:00:00Z",
            "tags": [],
        }
    ],
    "active_inbox_count": 0,
    "embedding_provider": "bm25-only",
    "daemon": {
        "version": "0.1.0",
        "uptime_seconds": 1234,
        "db_path": "/tmp/test.db",
    },
    "next_recommended_action": "Project is healthy; use recall<query> for specific context",
}


class FakeClient:
    """Async mock satisfying the ``DaemonClient`` protocol.

    Returns canned data unless ``raise_on`` lists method names that should
    raise ``httpx.ConnectError`` instead — the simulated daemon-down case.
    """

    def __init__(self, *, raise_on: Optional[List[str]] = None) -> None:
        self.raise_on = set(raise_on or [])

    def _maybe_raise(self, name: str) -> None:
        if name in self.raise_on:
            raise httpx.ConnectError("simulated daemon-down")

    async def health(self) -> Dict[str, Any]:
        self._maybe_raise("health")
        return {"status": "ok"}

    async def briefing(self, scope: Optional[str]) -> Dict[str, Any]:
        self._maybe_raise("briefing")
        return dict(_CANNED_BRIEFING)

    async def recall(self, query: str, scope: str, limit: int = 10) -> Dict[str, Any]:
        self._maybe_raise("recall")
        return {
            "hits": [
                {
                    "id": "frag-1",
                    "type": "decision",
                    "score": 0.91,
                    "content": "Pin Python to >=3.9 for now",
                    "created_by_tool": "claude_code",
                    "scope": scope,
                }
            ]
        }

    async def list_clients(self) -> List[Dict[str, Any]]:
        self._maybe_raise("list_clients")
        return [
            {"id": "claude_code", "display_name": "Claude Code",
             "detected": True, "connected": True},
            {"id": "cursor", "display_name": "Cursor",
             "detected": True, "connected": False},
        ]

    async def list_inbox(self, scope: Optional[str], limit: int = 50) -> List[Dict[str, Any]]:
        self._maybe_raise("list_inbox")
        return [
            {
                "id": "cand-aaaaaaaa",
                "type": "decision",
                "confidence": 0.72,
                "source_tool": "git_watcher",
                "content": "switch to launchd for daemon persistence",
            }
        ]

    async def approve_candidate(self, candidate_id: str) -> Dict[str, Any]:
        return {"returncode": 0, "stdout": "", "stderr": ""}

    async def reject_candidate(self, candidate_id: str) -> Dict[str, Any]:
        return {"returncode": 0, "stdout": "", "stderr": ""}

    async def read_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        self._maybe_raise("read_events")
        return [
            {"ts": "2026-05-13T10:00:00Z", "event": "recall",
             "scope": "project:test-scope", "details": {"query": "x", "hits": 3}},
        ]

    async def close(self) -> None:  # pragma: no cover - trivial
        pass


def _make_app(*, raise_on: Optional[List[str]] = None) -> SkeinApp:
    """Build a SkeinApp wired to a FakeClient with no daemon round trips."""
    return SkeinApp(
        scope="project:test-scope",
        client_factory=lambda: FakeClient(raise_on=raise_on),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_app_imports_cleanly() -> None:
    """The simplest possible smoke — the module imports + class exists."""
    from skein.tui.app import SkeinApp as _SkeinApp
    assert _SkeinApp is SkeinApp


@pytest.mark.asyncio
async def test_app_renders_briefing_offline() -> None:
    """Briefing tab renders canned data without touching a daemon."""
    app = _make_app()
    async with app.run_test() as pilot:
        # Allow background workers (refresh_health, refresh_briefing) to settle.
        await pilot.pause()
        await pilot.pause()
        # Header reflects the injected scope.
        from skein.tui.widgets.health_dot import HealthHeader
        header = app.query_one(HealthHeader)
        assert header.scope == "project:test-scope"
        # Briefing pane shows the canned fragment_total.
        from textual.widgets import Static
        stats = app.query_one("#briefing-stats", Static)
        text = stats.render()
        rendered = str(text)
        assert "fragment_total" in rendered
        # Daemon banner is hidden when the briefing call succeeds.
        banner = app.query_one("#daemon-banner", Static)
        assert "-hidden" in banner.classes


@pytest.mark.asyncio
async def test_app_handles_daemon_down() -> None:
    """When briefing() raises ConnectError, banner appears + app stays alive."""
    app = _make_app(raise_on=["briefing"])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        from textual.widgets import Static
        banner = app.query_one("#daemon-banner", Static)
        # Banner is visible (the -hidden class is removed).
        assert "-hidden" not in banner.classes
        rendered = str(banner.render())
        assert "daemon not running" in rendered.lower()
        # Health header reflects the failure.
        from skein.tui.widgets.health_dot import HealthHeader
        header = app.query_one(HealthHeader)
        assert header.healthy is False


@pytest.mark.asyncio
async def test_help_modal_opens_and_dismisses() -> None:
    """Press '?' to open help, escape to dismiss; modal mounts a help-dialog."""
    from skein.tui.screens.help_modal import HelpModal

    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("?")
        await pilot.pause()
        assert isinstance(app.screen, HelpModal)
        await pilot.press("escape")
        await pilot.pause()
        # Back on the main screen.
        assert not isinstance(app.screen, HelpModal)


@pytest.mark.asyncio
async def test_tab_navigation() -> None:
    """Number chords switch tabs; ``2`` lands on Fragments."""
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual.widgets import TabbedContent
        tabs = app.query_one(TabbedContent)
        # Default is briefing.
        assert tabs.active == "briefing"
        await pilot.press("2")
        await pilot.pause()
        assert tabs.active == "fragments"
        await pilot.press("5")
        await pilot.pause()
        assert tabs.active == "clients"


@pytest.mark.asyncio
async def test_quit_chord_exits_cleanly() -> None:
    """Pressing 'q' exits without raising."""
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    # App finished without raising; nothing more to assert.
