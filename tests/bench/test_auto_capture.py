"""Pytest gate for auto-capture (decision detection) quality.

Skein's current noise filter has known false positives (``test:`` type
commits and "Initial commit" — see ``commits.json``). We allow precision
down to 0.75 and recall ≥ 0.9. Tighten as the filter improves.
"""
from __future__ import annotations

from bench.scenarios.auto_capture import measure_auto_capture_quality


def test_decision_capture_precision_floor(ephemeral_adapter):
    result = measure_auto_capture_quality(ephemeral_adapter)
    if result.status == "skipped":
        return  # tool doesn't support git capture
    assert result.metrics["precision"] >= 0.75, (
        f"precision={result.metrics['precision']:.2f} below 0.75 — "
        f"the noise filter is letting too much through. "
        f"Disagreements: {result.notes.get('disagreements')}"
    )


def test_decision_capture_recall_floor(ephemeral_adapter):
    result = measure_auto_capture_quality(ephemeral_adapter)
    if result.status == "skipped":
        return
    assert result.metrics["recall"] >= 0.9, (
        f"recall={result.metrics['recall']:.2f} below 0.9 — "
        f"real decisions are being filtered as noise. "
        f"Disagreements: {result.notes.get('disagreements')}"
    )
