"""Promote passively-extracted facts/candidates into the fragment store.

iter 14.1 / 14.2: the bridge between the passive watchers (code scanner,
transcript extractor) and Skein's existing fragments + extraction_candidates
tables. High-confidence findings auto-promote; medium-confidence findings
land in the review queue; low-confidence is discarded.

Topic-keyed dedup (iter 18): every ``ScannedFact`` from the code scanner
carries a stable ``topic_key`` (e.g. ``tests-layout``, ``python-dep:fastapi``)
that identifies *which fact slot* it fills. When a new emission with the same
topic_key arrives, the old fragment is superseded rather than duplicated.
Pre-keyed legacy fragments are matched via a content-stem fingerprint so the
first scan after upgrade also consolidates the buggy duplicates that the
plain content-equality dedup missed.

This module is deliberately tiny so the watchers stay declarative ("here
are facts I found") and don't have to know about the storage layer.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .events import log_event
from .models import CommitCreate, Fragment, FragmentCreate, FragmentUpdate
from .scanner import ScannedFact, classify
from .storage import ConflictError

logger = logging.getLogger("skein.passive")


@dataclass
class PromoteResult:
    auto_promoted: int = 0
    queued: int = 0
    discarded: int = 0
    duplicate: int = 0
    superseded: int = 0  # iter 18: fragments retired during topic-keyed replace


def promote_scanned_facts(
    facts: Iterable[ScannedFact],
    *,
    storage,
    provider,
    scope_id: str,
    owner_id: str,
    source_tool: str,
) -> PromoteResult:
    """Route each ``ScannedFact`` into fragments or extraction_candidates.

    - confidence ≥ AUTO_PROMOTE_THRESHOLD → create a real fragment now
      (idempotent via topic_key + content-hash check; supersedes the old
      fragment when content changes for an existing topic)
    - DISCARD_THRESHOLD ≤ confidence < AUTO_PROMOTE_THRESHOLD → enqueue for
      ``skein inbox`` review
    - confidence < DISCARD_THRESHOLD → drop on the floor

    Returns a small result struct for the CLI to render a summary line.
    """
    from .embeddings import vec_to_bytes

    result = PromoteResult()
    facts = list(facts)
    if not facts:
        return result

    auto_facts = [f for f in facts if classify(f) == "auto"]
    queue_facts = [f for f in facts if classify(f) == "queue"]
    result.discarded = sum(1 for f in facts if classify(f) == "discard")

    # ---- Auto-promote: write fragments directly ----
    if auto_facts:
        existing = _load_existing_passive_fragments(storage, scope_id, source_tool)
        by_topic, by_stem = _index_existing(existing)

        commit = storage.create_commit(CommitCreate(
            author_id=owner_id, scope_id=scope_id,
            message=f"[{source_tool}] {len(auto_facts)} auto-extracted fact(s)",
        ))
        added_ids: list[str] = []
        for f in auto_facts:
            matches = _find_matches(f, by_topic, by_stem)
            if matches and all(m.content == f.content for m in matches):
                # All matching fragments already carry this exact content.
                # No-op for the most recent; retire stragglers from a
                # pre-fix DB where the bug allowed duplicates to coexist.
                _retire_duplicate_stragglers(storage, matches, source_tool)
                result.duplicate += 1
                continue

            embedding_bytes = None
            try:
                vec = provider.embed_one(f.content)
                embedding_bytes = vec_to_bytes(vec)
            except Exception:
                pass

            supersedes_id: str | None = None
            stale_targets: list[Fragment] = []
            if matches:
                # Pick the most recent match for the supersede chain pointer.
                most_recent = max(matches, key=lambda m: m.created_at)
                supersedes_id = most_recent.id
                stale_targets = list(matches)

            metadata: dict[str, object] = {}
            if f.topic_key:
                metadata["topic_key"] = f.topic_key

            frag = storage.create_fragment(
                FragmentCreate(
                    content=f.content,
                    type=f.type,
                    scope_id=scope_id,
                    owner_id=owner_id,
                    territory=f.territory,
                    tags=f.tags,
                    metadata=metadata,
                    created_by_tool=source_tool,
                    extraction_method=source_tool,
                    extraction_confidence=f.confidence,
                    supersedes_fragment_id=supersedes_id,
                ),
                commit_id=commit.id, embedding=embedding_bytes,
            )
            added_ids.append(frag.id)
            result.auto_promoted += 1
            log_event(
                "passive_auto_promote", scope=None,
                source_tool=source_tool,
                fragment_id=frag.id, preview=f.content[:80],
                confidence=f.confidence,
                topic_key=f.topic_key,
            )

            # Mark every matched fragment stale. ``create_fragment`` already
            # set ``superseded_by_fragment_id`` on the one pointed at by
            # ``supersedes_id``; the others are duplicate-cleanup casualties.
            for old in stale_targets:
                ok = _mark_stale(
                    storage, old.id,
                    reason=f"superseded by {frag.id}",
                )
                if ok:
                    result.superseded += 1
                    log_event(
                        "passive_supersede", scope=None,
                        source_tool=source_tool,
                        old_fragment_id=old.id, new_fragment_id=frag.id,
                        topic_key=f.topic_key,
                    )

        if added_ids:
            storage._conn.execute(
                "UPDATE commits SET fragments_added = ? WHERE id = ?",
                ("[" + ",".join(f'"{i}"' for i in added_ids) + "]", commit.id),
            )

    # ---- Queue: write to extraction_candidates ----
    for f in queue_facts:
        cid = storage.add_extraction_candidate(
            scope_id=scope_id, content=f.content, type=f.type,
            confidence=f.confidence, source_tool=source_tool,
            territory=f.territory, tags=f.tags,
            source_file=f.source_file,
        )
        if cid:
            result.queued += 1
            log_event(
                "passive_queue", scope=None, candidate_id=cid,
                source_tool=source_tool, confidence=f.confidence,
                preview=f.content[:80],
            )
        else:
            result.duplicate += 1
    return result


# ---------------------------------------------------------------------------
# Topic-keyed lookup helpers
# ---------------------------------------------------------------------------


def _load_existing_passive_fragments(
    storage, scope_id: str, source_tool: str,
) -> list[Fragment]:
    """Pull live fragments from the same scope+tool for dedup."""
    from .storage import _row_to_fragment

    rows = storage._conn.execute(
        "SELECT * FROM fragments WHERE scope_id = ? AND created_by_tool = ? "
        "AND is_stale = 0",
        (scope_id, source_tool),
    ).fetchall()
    return [_row_to_fragment(r) for r in rows]


def _index_existing(
    fragments: list[Fragment],
) -> tuple[dict[str, list[Fragment]], dict[str, list[Fragment]]]:
    """Index existing fragments by topic_key and by content-stem.

    A single legacy fragment with no ``metadata.topic_key`` only appears in the
    stem index. A new-style fragment appears in both — the topic_key map is
    authoritative; the stem map is the migration-bridge fallback.
    """
    by_topic: dict[str, list[Fragment]] = {}
    by_stem: dict[str, list[Fragment]] = {}
    for f in fragments:
        tk = (f.metadata or {}).get("topic_key")
        if tk:
            by_topic.setdefault(tk, []).append(f)
        by_stem.setdefault(_content_stem(f.content), []).append(f)
    return by_topic, by_stem


def _find_matches(
    fact: ScannedFact,
    by_topic: dict[str, list[Fragment]],
    by_stem: dict[str, list[Fragment]],
) -> list[Fragment]:
    """Find existing fragments that fill the same fact slot as ``fact``.

    Prefers topic_key matches. Falls back to content-stem matches so the
    first scan after the iter-18 upgrade consolidates legacy un-keyed
    duplicates instead of stacking a third copy.

    Stem fallback only crosses *into* fragments that don't already declare
    a topic_key (legacy) or that declare the *same* topic_key. Otherwise
    two distinct topics whose contents share a stem (e.g. two different
    Dockerfile EXPOSE ports) would cross-supersede on every scan.
    """
    matches: dict[str, Fragment] = {}
    if fact.topic_key:
        for f in by_topic.get(fact.topic_key, []):
            matches[f.id] = f
    stem = _content_stem(fact.content)
    for f in by_stem.get(stem, []):
        existing_tk = (f.metadata or {}).get("topic_key")
        if existing_tk and existing_tk != fact.topic_key:
            continue
        matches[f.id] = f
    return list(matches.values())


# Digits in scanner output are the unstable part (file counts, version
# numbers, port numbers within a fact's content body). Stripping them
# yields a fingerprint that's stable across the kinds of mutations the
# scanner naturally observes.
_DIGITS = re.compile(r"\d+")


def _content_stem(s: str) -> str:
    return _DIGITS.sub("N", s).strip().lower()


def _retire_duplicate_stragglers(
    storage, matches: list[Fragment], source_tool: str,
) -> None:
    """When several legacy fragments share the same content as the new fact,
    keep the most recent and mark the rest stale. No new fragment is created.
    """
    if len(matches) <= 1:
        return
    keeper = max(matches, key=lambda m: m.created_at)
    for m in matches:
        if m.id == keeper.id:
            continue
        _mark_stale(storage, m.id, reason=f"duplicate of {keeper.id}")
        log_event(
            "passive_dedup_cleanup", scope=None,
            source_tool=source_tool,
            stale_fragment_id=m.id, keeper_fragment_id=keeper.id,
        )


def _mark_stale(storage, frag_id: str, *, reason: str) -> bool:
    """Mark a fragment stale with a fresh-version read for the OCC check.

    Returns True on success, False if the fragment was modified between
    the read and write (we just leave it live; another scan will retry).
    """
    current = storage.get_fragment(frag_id)
    if current is None or current.is_stale:
        return False
    try:
        storage.update_fragment(frag_id, FragmentUpdate(
            is_stale=True, stale_reason=reason,
            expected_version=current.version,
        ))
        return True
    except ConflictError:
        logger.debug("OCC conflict marking %s stale — leaving live", frag_id)
        return False
