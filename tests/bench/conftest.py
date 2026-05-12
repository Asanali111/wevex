"""Fixtures for the bench pytest layer."""
from __future__ import annotations

import pytest

from bench.adapters.skein_ephemeral import SkeinEphemeralAdapter


@pytest.fixture
def ephemeral_adapter():
    """Fresh in-process Skein on a tmp DB. Closed at teardown."""
    a = SkeinEphemeralAdapter()
    try:
        a.ensure_scope("project:bench")
        yield a
    finally:
        a.close()
