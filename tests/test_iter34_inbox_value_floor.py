"""Iter 34 antibodies — value-aware inbox early-reject.

The inbox auto-approve sweep used to gate purely on confidence + age:
high-confidence → promote, old-and-low → reject, everything else → keep.
That left a "limbo" band (transcript-extracted mid-sentence chat noise
at confidence ~0.62) sitting pending for the full 14-day reject window
before draining.

Iter 34 adds a per-source-tool value-floor gate so transcript noise is
rejected on the first sweep, while structured passive scanners
(code-scanner, docs-watcher) keep their existing confidence-only path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from skein.server import _inbox_sweep_decision


def _ts(days_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


PROMOTE = 0.85
REJECT_CUTOFF = datetime.now(timezone.utc) - timedelta(days=14)
FLOOR = 0.35
PREFIX = "transcript"


class TestPromotionPathUnchanged:
    """High-confidence candidates still promote regardless of source_tool."""

    def test_promote_high_confidence_transcript(self):
        d = _inbox_sweep_decision(
            confidence=0.90,
            source_tool="transcript-claude",
            type_="decision",
            content="Decided to use fastembed as the default provider.",
            created_at_ts=_ts(1),
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        assert d == "promote"

    def test_promote_high_confidence_passive(self):
        d = _inbox_sweep_decision(
            confidence=0.95,
            source_tool="code-scanner",
            type_="fact",
            content="Uses Python package fastembed (declared: fastembed>=0.3.0).",
            created_at_ts=_ts(1),
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        assert d == "promote"


class TestValueRejectFiresForTranscriptNoise:
    """The 141-pending data shape: transcript-* + 0.62 confidence + short
    mid-sentence content sits at value≈0.20-0.30. These now get rejected
    on the first sweep rather than waiting 14 days."""

    def test_transcript_short_observation_rejected_immediately(self):
        # Real example from the 141: "approved" — value=0.20, would have
        # sat pending until day 14.
        d = _inbox_sweep_decision(
            confidence=0.62,
            source_tool="transcript-claude",
            type_="observation",
            content="approved",
            created_at_ts=_ts(0.5),  # fresh
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        assert d == "value_reject"

    def test_transcript_mid_sentence_rejected_immediately(self):
        # Real example: "the demo is the entire marketing strategy until
        # you have 100+ users" — broken mid-sentence, low value.
        d = _inbox_sweep_decision(
            confidence=0.62,
            source_tool="transcript-claude",
            type_="observation",
            content=(
                "the demo is the entire marketing strategy until you have "
                "100+ users"
            ),
            created_at_ts=_ts(0.5),
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        assert d == "value_reject"


class TestValueRejectSpaSesStructuredContent:
    """Borderline transcript-extracted content that lands above the 0.35
    floor — i.e., type-bonus carries it across despite the transcript
    provenance prior — keeps the existing keep-then-reject-on-age path.

    The empirical "Preference: call recall first before responding"
    candidate from the iter-34 corpus sits at exactly 0.40 (transcript
    base 0.30 + type bonus 0.10 + no content penalties because density
    clears the floor for short prefixed lines).
    """

    def test_preference_prefix_not_value_rejected(self):
        d = _inbox_sweep_decision(
            confidence=0.72,
            source_tool="transcript-claude",
            type_="preference",
            content="Preference: call recall first before responding",
            created_at_ts=_ts(0.5),
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        # Confidence below 0.85, value 0.40 ≥ floor, age below cutoff
        # → keep for normal human review or 14-day age reject.
        assert d == "keep"


class TestPassiveScannersUntouched:
    """code-scanner and docs-watcher don't match the floor_prefix, so
    even a low-value candidate from them stays on the age-based path."""

    def test_code_scanner_low_value_keeps(self):
        d = _inbox_sweep_decision(
            confidence=0.62,
            source_tool="code-scanner",
            type_="observation",
            content="Edit on /tmp/foo",  # this is value=0.05 (rubric floor)
            created_at_ts=_ts(0.5),
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        # Not transcript → no value-reject → not old → keep.
        assert d == "keep"


class TestAgeBasedRejectStillFires:
    """Old transcript content that didn't trigger the value-floor reject
    on earlier sweeps (e.g., value-floor was 0 before iter 34) still
    rejects on the age path."""

    def test_old_above_floor_transcript_rejects_by_age(self):
        # Pick content that clears the 0.35 floor (so the value-reject
        # path is bypassed) but is also old, to exercise the age branch.
        old = datetime.now(timezone.utc) - timedelta(days=30)
        d = _inbox_sweep_decision(
            confidence=0.72,
            source_tool="transcript-claude",
            type_="preference",
            content="Preference: call recall first before responding",
            created_at_ts=old,
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        assert d == "reject"

    def test_old_passive_low_value_rejects_by_age(self):
        # Passive scanner content isn't subject to the value-floor gate
        # (only transcript-* prefix triggers it), so it falls through to
        # the age path when stale.
        old = datetime.now(timezone.utc) - timedelta(days=30)
        d = _inbox_sweep_decision(
            confidence=0.62,
            source_tool="code-scanner",
            type_="observation",
            content="Edit on /tmp/foo",
            created_at_ts=old,
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix=PREFIX,
        )
        assert d == "reject"


class TestDisabledViaConfig:
    """Setting value_floor=0 turns off iter-34 behaviour entirely."""

    def test_zero_floor_disables_value_reject(self):
        d = _inbox_sweep_decision(
            confidence=0.62,
            source_tool="transcript-claude",
            type_="observation",
            content="approved",
            created_at_ts=_ts(0.5),
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=0.0,
            floor_prefix=PREFIX,
        )
        assert d == "keep"

    def test_empty_prefix_disables_value_reject(self):
        d = _inbox_sweep_decision(
            confidence=0.62,
            source_tool="transcript-claude",
            type_="observation",
            content="approved",
            created_at_ts=_ts(0.5),
            promote_threshold=PROMOTE,
            reject_cutoff=REJECT_CUTOFF,
            value_floor=FLOOR,
            floor_prefix="",
        )
        assert d == "keep"
