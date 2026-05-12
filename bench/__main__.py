"""CLI entry: ``python -m bench --adapter skein --mode ephemeral`` or ``--mode live``.

The live mode is read-only against the daemon at 127.0.0.1:8765 (skips
mutable scenarios). Ephemeral mode spins up a fresh in-process Skein on a
temp SQLite DB and runs every scenario.
"""
from __future__ import annotations

import argparse
import json
import sys

from .adapter import ReadOnlyAdapter
from .reporter import render_markdown
from .runner import run


def _build_adapter(name: str, mode: str) -> ReadOnlyAdapter:
    if name == "skein":
        if mode == "live":
            from .adapters.skein_live import SkeinLiveAdapter
            return SkeinLiveAdapter()
        if mode == "ephemeral":
            from .adapters.skein_ephemeral import SkeinEphemeralAdapter
            return SkeinEphemeralAdapter()
        raise SystemExit(f"unknown mode: {mode}")
    raise SystemExit(f"unknown adapter: {name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench")
    p.add_argument("--adapter", default="skein", help="adapter name (default: skein)")
    p.add_argument("--mode", choices=["live", "ephemeral"], default="ephemeral",
                   help="live = read-only against running daemon; "
                        "ephemeral = fresh in-process daemon")
    p.add_argument("--scope", default="project:bench",
                   help="scope to use; for live mode pass an existing scope handle")
    p.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--no-correctness", action="store_true")
    p.add_argument("--no-auto-capture", action="store_true")
    args = p.parse_args(argv)

    adapter = _build_adapter(args.adapter, args.mode)
    try:
        report = run(
            adapter, scope=args.scope,
            include_quality=not args.no_quality,
            include_correctness=not args.no_correctness,
            include_auto_capture=not args.no_auto_capture,
        )
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_markdown(report))

    return 0 if report.overall_status in ("pass", "warn") else 1


if __name__ == "__main__":
    sys.exit(main())
