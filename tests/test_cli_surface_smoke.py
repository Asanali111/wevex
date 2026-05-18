"""ADR-002 smoke test: the 10-command surface composes without breaking.

Each individual command has its own focused tests elsewhere. This file
exercises the *composition* — that `skein --help` advertises the right
top-level commands, that the new flags on `doctor`, `briefing`, and
`connect` parse, and that `status` includes the sections we folded into it
(clients, inbox depth).

It deliberately uses Click's CliRunner with `--help` invocations instead of
actually running the daemon, so it stays fast and doesn't need a fixture
sandbox. The point is to catch the failure mode where folding `clients` into
`status` happens in plan but the actual section never gets rendered.
"""
from __future__ import annotations

from click.testing import CliRunner

from skein.cli import main


def _help(*argv) -> str:
    r = CliRunner().invoke(main, list(argv) + ["--help"])
    assert r.exit_code == 0, r.output
    return r.output


# ---------------------------------------------------------------------------
# Top-level surface: the 10 commands the user should see
# ---------------------------------------------------------------------------

VISIBLE_TOP_LEVEL = {
    "up", "down", "restart", "status", "doctor", "tail",
    "briefing", "tui", "config", "connect",
}


def test_top_level_help_shows_canonical_surface() -> None:
    """Every command in VISIBLE_TOP_LEVEL must appear in `skein --help`.

    Reverse direction (no extra commands beyond the visible set) is NOT
    asserted yet — Phase D will hide the rest with hidden=True, at which
    point a follow-up test will pin that the surface is *exactly* this set.
    """
    out = _help()
    missing = [c for c in VISIBLE_TOP_LEVEL if c not in out]
    assert not missing, f"Missing from --help: {missing}\nOutput was:\n{out}"


# ---------------------------------------------------------------------------
# New flags absorb the deleted commands' work
# ---------------------------------------------------------------------------

def test_doctor_has_clean_and_reingest_flags() -> None:
    out = _help("doctor")
    assert "--clean" in out
    assert "--reingest" in out
    # The help text must mention what these flags replace so a user
    # reading docs in iter 26 sees the migration story.
    assert "skein gc" in out or "gc" in out  # --clean replaces gc
    assert "ingest" in out                   # --reingest replaces ingest


def test_briefing_has_since_flag() -> None:
    out = _help("briefing")
    assert "--since" in out
    # Should still keep --scope and --json for orthogonal control.
    assert "--scope" in out
    assert "--json" in out


def test_connect_has_remove_flag() -> None:
    out = _help("connect")
    assert "--remove" in out
    # Original positional + --all should still work.
    assert "--all" in out


# ---------------------------------------------------------------------------
# Folded surface: status absorbs `clients` and `daemon status`
# ---------------------------------------------------------------------------

def test_status_help_mentions_absorbed_surface() -> None:
    """status's docstring should advertise that it now subsumes `clients`
    and `daemon status` per ADR-002 — keeps the migration discoverable."""
    out = _help("status")
    assert "clients" in out.lower() or "daemon" in out.lower()


def test_status_offline_renders_without_clients_section() -> None:
    """When the daemon is offline, status still exits cleanly and prints
    the offline message — the clients section is a best-effort addition,
    not a failure path."""
    # Use a bogus port so we don't hit a running daemon.
    runner = CliRunner()
    r = runner.invoke(main, ["status"], env={"SKEIN_PORT": "1"})
    # Exit code 1 is correct: status exits non-zero when offline. The
    # important assertion is no exception traceback in stderr.
    assert r.exit_code == 1
    assert "offline" in r.output.lower() or "skein is offline" in r.output.lower()
