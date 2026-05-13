"""Scenarios — tool-agnostic measurements run against an adapter.

A scenario produces a ``ScenarioResult`` with named metrics and a status.
The runner aggregates results and the reporter formats them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

Status = Literal["pass", "warn", "fail", "skipped", "error"]


@dataclass
class ScenarioResult:
    name: str
    category: str
    status: Status = "pass"
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "category": self.category, "status": self.status,
            "metrics": self.metrics, "reason": self.reason, "notes": self.notes,
        }
