"""Render a BenchmarkReport as a markdown report."""
from __future__ import annotations

from collections.abc import Iterable

from .runner import BenchmarkReport
from .scenarios import ScenarioResult

_STATUS_GLYPH = {
    "pass": "PASS", "warn": "WARN", "fail": "FAIL",
    "skipped": "skip", "error": "ERR ",
}


def _fmt_num(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def render_markdown(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append(f"# Context-bus benchmark report — `{report.adapter_name}`")
    lines.append("")
    lines.append(f"**Overall status:** `{report.overall_status.upper()}`  ")
    h = report.health
    lines.append(
        f"**Tool health:** {h.get('fragment_count', 0)} fragments · "
        f"{h.get('chunk_count', 0)} chunks · "
        f"{h.get('scope_count', 0)} scopes · version `{h.get('version', '?')}`"
    )
    lines.append("")
    lines.append("> Cross-tool round-trip is measured at the API level, not "
                 "end-to-end through real IDEs. A passing benchmark proves the "
                 "daemon serves correct data; it does not prove that Claude / "
                 "Cursor / Codex actually surface it.")
    lines.append("")

    by_cat = _group_by_category(report.scenarios)
    for cat, scenarios in by_cat.items():
        lines.append(f"## {cat.title()}")
        lines.append("")
        lines.extend(_render_scenarios(scenarios, report))
        lines.append("")

    fail = [s.name for s in report.scenarios if s.status == "fail"]
    warn = [s.name for s in report.scenarios if s.status == "warn"]
    if fail:
        lines.append("### Failing scenarios")
        lines.extend(f"- `{name}`" for name in fail)
        lines.append("")
    if warn:
        lines.append("### Warning scenarios")
        lines.extend(f"- `{name}`" for name in warn)
        lines.append("")

    return "\n".join(lines)


def _group_by_category(scenarios: Iterable[ScenarioResult]) -> dict:
    out: dict = {}
    for s in scenarios:
        out.setdefault(s.category, []).append(s)
    return out


def _render_scenarios(scenarios: list[ScenarioResult], report: BenchmarkReport) -> list[str]:
    lines: list[str] = []
    for s in scenarios:
        glyph = _STATUS_GLYPH.get(s.status, s.status)
        header = f"### [{glyph}] `{s.name}`"
        if s.reason:
            header += f"  — {s.reason}"
        lines.append(header)
        if s.status == "skipped":
            lines.append("")
            continue
        # Metrics table
        budget = report.budget_evaluations.get(s.name, {})
        if s.metrics:
            lines.append("")
            lines.append("| metric | observed | budget |")
            lines.append("|---|---|---|")
            for k, v in s.metrics.items():
                b = budget.get(k)
                if b:
                    op = b["op"]
                    thr = _fmt_num(b["threshold"])
                    bcell = f"{op} {thr} {'OK' if b['ok'] else 'MISS'}"
                else:
                    bcell = "—"
                lines.append(f"| `{k}` | {_fmt_num(v)} | {bcell} |")
            lines.append("")
        # Notes — only render disagreements / per-query rows if small.
        if s.notes:
            for key, val in s.notes.items():
                if isinstance(val, list) and val and len(val) <= 25:
                    lines.append(f"<details><summary>{key} ({len(val)})</summary>")
                    lines.append("")
                    for item in val:
                        lines.append(f"- `{item}`")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
    return lines
