"""Compute a fragment's recall-time value (Q-05 phases 1+2).

The store accumulates session noise: tool-event observations like ``Edit on
/path/file.py``, dep-facts the scanner re-emits on every restart, transcript
extractions of mid-conversation reasoning. Users escape this by reopening
their chat, but Skein has no equivalent reset — so the value of a fragment
to *future* sessions has to be estimated at write-time and applied as a
boost/penalty during recall.

This module is a deterministic, no-LLM scorer. Three signal families combine
into a single ``value`` in [0.05, 1.0]:

* Provenance prior — the *origin* of the fragment is the strongest cheap
  signal. A human typing ``skein remember`` is near-certainly worth
  surfacing; a watcher firing on a tool event is near-certainly not.
* Type prior — ``decision``/``requirement``/``procedure`` carry durable
  intent; ``observation``/``conversation`` are typically activity logs.
* Content score — regex rubrics for known noise patterns ("Edit on …"),
  information-density floor, length normalisation.

Behavioural signal (recall hits, supersede position, decay) lands in phase 3
and updates the column in place. The value here is the *initial* value.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Provenance prior — base value by (extraction_method, created_by_tool, …)
# ---------------------------------------------------------------------------

# Source-tool labels that the watchers and scanners stamp onto fragments.
# Keeping these in one place so any extractor change shows up here.
_PASSIVE_TOOLS = {"code-scanner", "scanner", "docs-watcher"}
_TRANSCRIPT_TOOLS = {
    "transcript-claude", "transcript-cursor", "transcript-codex",
}
_INBOX_AUTO_PREFIX = "inbox-auto-approve"
_INBOX_MANUAL_PREFIX = "inbox-approve"


def _provenance_base_value(
    *,
    extraction_method: str,
    created_by_tool: Optional[str],
    metadata: Mapping[str, Any],
) -> float:
    """Return the base value from the fragment's origin."""
    em = (extraction_method or "explicit").lower()
    tool = (created_by_tool or "").lower()

    # Inbox path — the auto-approve flow stamps ``extraction_method`` with
    # the source tool's label, not "inbox", so detect via metadata (set by
    # ``_promote_candidate``) or the commit prefix.
    if metadata.get("promoted_via") == _INBOX_AUTO_PREFIX:
        return 0.55
    if metadata.get("promoted_via") == _INBOX_MANUAL_PREFIX:
        return 0.65

    # Explicit user/agent writes
    if em == "explicit":
        # ``note_decision`` populates ``alternatives`` + ``rationale`` in the
        # fragment body itself, but the MCP server also drops a marker into
        # ``metadata`` so we can detect structured decisions cheaply.
        if metadata.get("has_alternatives") or metadata.get("structured_decision"):
            return 0.90
        # CLI-typed `skein remember` arrives without a tool label — the
        # MCP path stamps "claude-code" / "cursor" / etc.
        if tool in ("", "cli", "skein-cli", "human"):
            return 1.00
        return 0.70

    # Passive code/docs scanner
    if em in ("code-scan", "scanner") or tool in _PASSIVE_TOOLS:
        return 0.35

    # Transcript-extracted
    if em.startswith("transcript") or tool in _TRANSCRIPT_TOOLS:
        return 0.30

    # Tool-event observations (PostToolUse hook used to write these; now a
    # no-op, but legacy rows still exist and the pattern matches reliably).
    if em in ("tool-event", "hook-observation"):
        return 0.10

    # Unknown — neutral default.
    return 0.40


# ---------------------------------------------------------------------------
# Type prior
# ---------------------------------------------------------------------------

_TYPE_ADJUSTMENT: dict[str, float] = {
    "decision":      0.10,
    "requirement":   0.10,
    "procedure":     0.10,
    "preference":    0.10,
    "fact":          0.00,
    "state":        -0.10,
    "observation":  -0.20,
    "conversation": -0.20,
}


# ---------------------------------------------------------------------------
# Content score — cheap regex/heuristic adjustments (phase 2)
# ---------------------------------------------------------------------------

# Tool-event activity-log shapes the PostToolUse hook used to emit. The hook
# is a no-op now (iter 11) but legacy fragments and a few stray paths still
# produce content of these shapes. Penalise hard.
_TOOL_EVENT_RE = re.compile(
    r"^(Edit|Write|Read|Bash|Grep|Glob|Task|MultiEdit|NotebookEdit) on "
)

# A "specific" token carries information: a path, a number, an identifier,
# or a proper-cased multi-letter word. Stopwords and bare verbs don't.
# Fused into one compiled regex so the density check is ~10× faster than
# the per-character ``any()`` walk it replaced — matters because this is on
# the create_fragment hot path and was blowing the bench p95 budget.
_SPECIFIC_TOKEN_RE = re.compile(
    r"[/\\.\d_]"          # path char, digit, or underscore anywhere
    r"|^[A-Z].*[a-z]",    # leading-uppercase + at-least-one-lowercase (proper-cased)
)


def _is_specific_token(t: str) -> bool:
    if len(t) < 3:
        return False
    return _SPECIFIC_TOKEN_RE.search(t) is not None


def _content_score_adjustment(content: str) -> float:
    """Sum of cheap content-based adjustments. Range roughly [-0.45, +0.0].

    Negative-only by design — content can pull value down but not lift it
    above what provenance + type already justified. That keeps the
    provenance prior as the load-bearing signal.
    """
    adj = 0.0

    # Filler-pattern penalty: activity-log shapes.
    if _TOOL_EVENT_RE.match(content):
        adj -= 0.30

    # Length penalty: extremes are usually low-information.
    n = len(content)
    if n < 20:
        adj -= 0.05
    elif n > 1500:
        adj -= 0.05

    # Density floor: too few specific tokens means the fragment is mostly
    # filler ("the", "is", "we", "should", …). Skip the check on very short
    # content where the ratio is noisy.
    if n >= 40:
        tokens = content.split()
        if tokens:
            specific = sum(1 for t in tokens if _is_specific_token(t))
            density = specific / len(tokens)
            if density < 0.10:
                adj -= 0.10

    return adj


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

VALUE_FLOOR = 0.05
VALUE_CEILING = 1.0


def compute_fragment_value(
    *,
    type: str,
    content: str,
    extraction_method: str = "explicit",
    created_by_tool: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> float:
    """Initial value for a fragment, in ``[VALUE_FLOOR, VALUE_CEILING]``.

    Deterministic, side-effect-free, ~microsecond runtime. Called once at
    fragment creation; the column it writes to can be updated later by the
    phase-3 telemetry path without touching this function.
    """
    md = metadata or {}
    base = _provenance_base_value(
        extraction_method=extraction_method,
        created_by_tool=created_by_tool,
        metadata=md,
    )
    type_adj = _TYPE_ADJUSTMENT.get(type, 0.0)
    content_adj = _content_score_adjustment(content)

    value = base + type_adj + content_adj
    if value < VALUE_FLOOR:
        return VALUE_FLOOR
    if value > VALUE_CEILING:
        return VALUE_CEILING
    return value
