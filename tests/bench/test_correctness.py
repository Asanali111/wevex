"""Pytest gate for the correctness invariants."""
from __future__ import annotations

from bench.scenarios.correctness import (
    check_fragment_typing,
    check_lease_lifecycle,
    check_scope_hierarchy,
)


def test_parent_scope_fragments_visible_from_child(ephemeral_adapter):
    result = check_scope_hierarchy(ephemeral_adapter)
    assert result.status == "pass", result.reason


def test_lease_lifecycle_round_trips(ephemeral_adapter):
    result = check_lease_lifecycle(ephemeral_adapter)
    assert result.status == "pass", result.reason


def test_unknown_fragment_type_rejected(ephemeral_adapter):
    result = check_fragment_typing(ephemeral_adapter)
    assert result.status == "pass", result.reason
