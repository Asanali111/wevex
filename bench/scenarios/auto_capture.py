"""Auto-capture quality — precision / recall of the decision detector.

For each labeled commit, ask the adapter "would you capture this?" and
compare to the ground-truth ``is_real_decision`` label. Reports precision,
recall, F1, plus per-commit disagreements so the user can inspect.

Why this matters: a context bus is only useful if the auto-extraction path
captures real decisions and rejects noise. A regex hitting every assistant
turn for the word "decided" will have great recall and terrible precision;
a strict allow-list will have great precision and terrible recall.
"""
from __future__ import annotations

from ..adapter import MutableAdapter
from ..corpus import commits
from ..scenarios import ScenarioResult


def measure_auto_capture_quality(adapter: MutableAdapter) -> ScenarioResult:
    if not adapter.supports_git_capture:
        return ScenarioResult(
            name="auto_capture_quality", category="quality", status="skipped",
            reason="adapter does not declare git-capture support",
        )

    try:
        # Probe the predicate once; raises NotImplementedError if unsupported.
        adapter.would_capture_commit("feat: probe", "")
    except NotImplementedError as e:
        return ScenarioResult(
            name="auto_capture_quality", category="quality", status="skipped",
            reason=str(e),
        )

    tp = fp = tn = fn = 0
    disagreements: list[dict] = []

    for c in commits():
        predicted = adapter.would_capture_commit(c["subject"], c.get("body", ""))
        actual = bool(c["is_real_decision"])
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
            disagreements.append({"id": c["id"], "kind": "false_positive",
                                  "subject": c["subject"]})
        elif not predicted and actual:
            fn += 1
            disagreements.append({"id": c["id"], "kind": "false_negative",
                                  "subject": c["subject"]})
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Thresholds: capture wants high precision (you don't want noise in the
    # bus). Floor at precision ≥ 0.7, recall ≥ 0.7.
    status = "pass"
    reason = ""
    if precision < 0.7:
        status, reason = "fail", f"precision={precision:.2f} below 0.7 floor"
    elif recall < 0.7:
        status, reason = "fail", f"recall={recall:.2f} below 0.7 floor"
    elif precision < 0.85 or recall < 0.85:
        status, reason = "warn", "precision or recall below 0.85 target"

    return ScenarioResult(
        name="auto_capture_quality",
        category="quality",
        status=status,
        metrics={
            "precision": precision, "recall": recall, "f1": f1,
            "true_positives": float(tp), "false_positives": float(fp),
            "true_negatives": float(tn), "false_negatives": float(fn),
            "n_commits": float(tp + fp + tn + fn),
        },
        reason=reason,
        notes={"disagreements": disagreements},
    )
