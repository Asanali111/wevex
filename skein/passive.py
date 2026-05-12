"""Promote passively-extracted facts/candidates into the fragment store.

iter 14.1 / 14.2: the bridge between the passive watchers (code scanner,
transcript extractor) and Skein's existing fragments + extraction_candidates
tables. High-confidence findings auto-promote; medium-confidence findings
land in the review queue; low-confidence is discarded.

This module is deliberately tiny so the watchers stay declarative ("here
are facts I found") and don't have to know about the storage layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .events import log_event
from .models import CommitCreate, FragmentCreate
from .scanner import AUTO_PROMOTE_THRESHOLD, DISCARD_THRESHOLD, ScannedFact, classify

logger = logging.getLogger("skein.passive")


@dataclass
class PromoteResult:
    auto_promoted: int = 0
    queued: int = 0
    discarded: int = 0
    duplicate: int = 0


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
      (idempotent via content-hash check)
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
        # Look up existing fragments in this scope from the same source_tool
        # to avoid re-creating identical facts on every scan.
        existing_contents = _existing_passive_contents(storage, scope_id, source_tool)

        commit = storage.create_commit(CommitCreate(
            author_id=owner_id, scope_id=scope_id,
            message=f"[{source_tool}] {len(auto_facts)} auto-extracted fact(s)",
        ))
        added_ids: List[str] = []
        for f in auto_facts:
            if f.content in existing_contents:
                result.duplicate += 1
                continue
            embedding_bytes = None
            try:
                vec = provider.embed_one(f.content)
                embedding_bytes = vec_to_bytes(vec)
            except Exception:
                pass
            frag = storage.create_fragment(
                FragmentCreate(
                    content=f.content,
                    type=f.type,
                    scope_id=scope_id,
                    owner_id=owner_id,
                    territory=f.territory,
                    tags=f.tags,
                    created_by_tool=source_tool,
                    extraction_method=source_tool,
                    extraction_confidence=f.confidence,
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


def _existing_passive_contents(storage, scope_id: str, source_tool: str) -> set:
    rows = storage._conn.execute(
        "SELECT content FROM fragments WHERE scope_id = ? AND created_by_tool = ? "
        "AND is_stale = 0",
        (scope_id, source_tool),
    ).fetchall()
    return {r["content"] for r in rows}
