"""Smoke tests for ``skein.ui`` helpers — render-without-crashing + pure
function correctness for ``home_relative`` / ``dot`` / ``mark``."""
from __future__ import annotations

from pathlib import Path

from skein import ui

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestDot:
    def test_known_states(self):
        for state in ["ok", "err", "warn", "info", "idle", "off"]:
            out = ui.dot(state)
            assert "●" in out or "○" in out

    def test_unknown_state_falls_back_to_ok(self):
        assert ui.dot("nonsense") == ui.dot("ok")


class TestMark:
    def test_known_states(self):
        for state in ["ok", "err", "warn", "step", "skip"]:
            out = ui.mark(state)
            assert any(g in out for g in ["✓", "✗", "⚠", "→", "·"])

    def test_unknown_state_falls_back_to_ok(self):
        assert ui.mark("nonsense") == ui.mark("ok")


class TestHomeRelative:
    def test_replaces_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        full = str(tmp_path / "Documents" / "thing")
        assert ui.home_relative(full) == "~/Documents/thing"

    def test_passthrough_for_unrelated_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert ui.home_relative("/etc/hosts") == "/etc/hosts"


# ---------------------------------------------------------------------------
# Render-without-crashing — every helper should print something
# ---------------------------------------------------------------------------

class TestRender:
    def test_header_runs(self, capsys):
        ui.header("Test", state="ok")
        captured = capsys.readouterr()
        assert "Test" in captured.out

    def test_section_runs(self, capsys):
        ui.section("Section")
        assert "Section" in capsys.readouterr().out

    def test_field_runs(self, capsys):
        ui.field("Label", "value")
        assert "Label" in capsys.readouterr().out

    def test_fields_auto_widths(self, capsys):
        ui.fields([("Short", "a"), ("LongLabel", "b")])
        out = capsys.readouterr().out
        assert "Short" in out
        assert "LongLabel" in out

    def test_status_list_runs(self, capsys):
        ui.status_list([
            ("ok", "id1", "Name 1", "note 1"),
            ("idle", "id2", "Name 2", "note 2"),
        ])
        out = capsys.readouterr().out
        assert "id1" in out and "id2" in out
        assert "Name 1" in out and "Name 2" in out

    def test_step_with_detail(self, capsys):
        ui.step("Did it", state="ok", detail="some/path")
        out = capsys.readouterr().out
        assert "Did it" in out and "some/path" in out

    def test_counter_line_drops_zeros(self, capsys):
        ui.counter_line([(0, "zero"), (3, "items"), (0, "another zero")])
        out = capsys.readouterr().out
        assert "items" in out
        assert "zero" not in out

    def test_counter_line_empty_prints_nothing(self, capsys):
        ui.counter_line([(0, "zero")])
        assert capsys.readouterr().out == ""

    def test_hint_runs(self, capsys):
        ui.hint("Try this")
        assert "Try this" in capsys.readouterr().out
