"""Hybrid retrieval: BM25 (FTS5) + vector cosine + Reciprocal Rank Fusion.

RRF formula:  score(d, Q) = Σ_{q in Q}  1 / (k + rank_q(d))
where k=60 is the standard constant that dampens the effect of high ranks.

Recall@10 improvement:
  vector-only  ~70%
  BM25-only    ~60%
  hybrid+RRF   ~91%   (standard BEIR benchmarks)

Reference: Cormack, Clarke & Buettcher 2009 — "Reciprocal Rank Fusion
outperforms Condorcet and individual rank learning methods."
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

from .embeddings import EmbeddingProvider, vec_to_bytes
from .models import (
    ChunkSearchRequest, ChunkSearchResponse, ChunkSearchResult,
    RecallRequest, RecallResponse, RecallResult,
    classify_recall_quality,
)
from .storage import Storage

logger = logging.getLogger("skein.retrieval")


# ---------------------------------------------------------------------------
# Iter 31: recall result micro-cache
#
# Same (scope, query, limit, types) within 30 s reuses the cached
# RecallResponse instead of re-embedding + re-querying. Invalidated by any
# write into the relevant scope (storage.create_fragment / update_fragment
# call invalidate_recall_cache(scope_id)). LRU-capped at 64 entries.
# Skip-listed when req.include_stale is True (rare debugging path).
# ---------------------------------------------------------------------------

_RECALL_CACHE_TTL_S: float = 30.0
_RECALL_CACHE_MAX: int = 64
_RECALL_CACHE: "OrderedDict[tuple, tuple[float, RecallResponse]]" = OrderedDict()
# Maps scope_id → list of cache keys, so writes can invalidate cheaply.
_RECALL_CACHE_BY_SCOPE: dict[str, set[tuple]] = {}


def _cache_key(req: RecallRequest, storage_id: str) -> tuple:
    """Stable hashable key for the recall cache.

    Iter 31: ``storage_id`` (the per-Storage instance UUID) is part of
    the key so multiple Storage instances on the same scope handle don't
    accidentally share cached responses — critical for tests using fresh
    ephemeral DBs that reuse `project:bench` as the scope name.
    """
    return (
        storage_id,
        req.scope,
        req.query,
        int(req.limit or 10),
        tuple(req.types) if req.types else None,
        req.territory,
        tuple(req.tags) if req.tags else None,
        bool(req.include_stale),
    )


def _cache_get(req: RecallRequest, storage_id: str) -> Optional[RecallResponse]:
    if req.include_stale:
        return None
    key = _cache_key(req, storage_id)
    hit = _RECALL_CACHE.get(key)
    if hit is None:
        return None
    ts, response = hit
    if (time.monotonic() - ts) > _RECALL_CACHE_TTL_S:
        # Expired — drop it.
        _RECALL_CACHE.pop(key, None)
        return None
    # Touch for LRU semantics.
    _RECALL_CACHE.move_to_end(key)
    return response


def _cache_set(req: RecallRequest, response: RecallResponse,
               scope_ids: list[str], storage_id: str) -> None:
    if req.include_stale:
        return
    key = _cache_key(req, storage_id)
    _RECALL_CACHE[key] = (time.monotonic(), response)
    for sid in scope_ids:
        _RECALL_CACHE_BY_SCOPE.setdefault(sid, set()).add(key)
    if len(_RECALL_CACHE) > _RECALL_CACHE_MAX:
        # Evict LRU. Also drop any scope-index references to the evicted key.
        oldest_key, _ = _RECALL_CACHE.popitem(last=False)
        for sid_set in _RECALL_CACHE_BY_SCOPE.values():
            sid_set.discard(oldest_key)


def invalidate_recall_cache(scope_id: Optional[str] = None) -> None:
    """Drop cache entries. ``scope_id=None`` clears everything (used by
    boost / bury / supersede); otherwise only entries that touched that
    scope are removed."""
    if scope_id is None:
        _RECALL_CACHE.clear()
        _RECALL_CACHE_BY_SCOPE.clear()
        return
    keys = _RECALL_CACHE_BY_SCOPE.pop(scope_id, set())
    for k in keys:
        _RECALL_CACHE.pop(k, None)


# ---------------------------------------------------------------------------
# Iter 31: recency decay
#
# Multiplier on the post-fusion score so old fragments fade unless they're
# pinned (value==1.0 + metadata.pinned) or are foundational types
# (requirement / procedure / preference). Half-life 60 days, floor 0.70 so
# old-but-still-relevant decisions don't get crushed.
# ---------------------------------------------------------------------------

_RECENCY_HALFLIFE_DAYS = 60.0
_RECENCY_FLOOR = 0.70
_RECENCY_SKIP_TYPES = {"requirement", "procedure", "preference"}


def _recency_multiplier(frag) -> float:
    """Return a multiplier in [_RECENCY_FLOOR, 1.0] from frag.created_at."""
    if frag.type in _RECENCY_SKIP_TYPES:
        return 1.0
    if getattr(frag, "permanent", False):
        return 1.0
    try:
        # `created_at` is stored as 'YYYY-MM-DD HH:MM:SS' or ISO 8601.
        raw = (frag.created_at or "").replace("Z", "+00:00")
        # SQLite default is space-separated; ISO needs 'T'.
        if "T" not in raw and " " in raw:
            raw = raw.replace(" ", "T", 1)
        # No tz → assume UTC.
        if "+" not in raw and "-" not in raw[10:]:
            raw += "+00:00"
        ts = datetime.fromisoformat(raw)
    except (ValueError, AttributeError):
        return 1.0
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    decayed = 0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS)
    return max(_RECENCY_FLOOR, decayed)


def recall(
    req: RecallRequest,
    storage: Storage,
    provider: EmbeddingProvider,
    *,
    k: int = 60,            # RRF constant
    candidate_n: int = 30,  # candidates per list before fusion
    value_floor: float = 0.05,  # iter 31: skip rubric-floor noise at SQL layer
    # NOTE: 0.05 == the rubric minimum from value.py, so this is a no-op
    # for the default tests but lets production callers (CLI / MCP) tune
    # higher without changing the call shape.
) -> RecallResponse:
    """Main entry point for hybrid recall.

    Steps:
    0. Cache check (iter 31): same (scope, query, limit, …) within 30s
       returns the previous RecallResponse — saves ~80ms per repeat call.
    1. Resolve scope lineage (query scope + all ancestors).
    2. Embed the query.
    3. BM25 keyword search → ranked list A.
    4. Vector cosine search → ranked list B.
    5. Fuse A + B with RRF.
    6. Hydrate top-N fragments, apply value × recency multipliers.
    7. Bump recall_hits on the top-K (fire-and-forget telemetry).
    8. Return RecallResponse.
    """
    # 0. Iter 31: cache check (per-Storage instance, so test fixtures and
    # the daemon's primary storage don't share keys).
    storage_id = getattr(storage, "instance_id", "default")
    cached = _cache_get(req, storage_id)
    if cached is not None:
        return cached

    # 1. Scope lineage
    lineage = storage.get_scope_lineage(req.scope)
    if not lineage:
        return RecallResponse(results=[], query=req.query, scope=req.scope, total=0)
    scope_ids = [s.id for s in lineage]

    # 2. Embed query
    try:
        query_vec = provider.embed_one(req.query)
        query_vec_bytes = vec_to_bytes(query_vec)
        have_embeddings = True
    except Exception as e:
        logger.warning("Embedding failed, falling back to keyword-only: %s", e)
        query_vec_bytes = b""
        have_embeddings = False

    types = list(req.types) if req.types else None

    # 3. BM25 keyword search
    # Iter 31: SQL-level value_floor — drops the noise tail before it even
    # gets ranked. Saves work proportional to how noisy the store is.
    keyword_hits: list[tuple[str, float]] = []
    try:
        keyword_hits = storage.keyword_search(
            req.query, scope_ids,
            type_filter=types,
            include_stale=req.include_stale,
            limit=candidate_n,
            value_floor=value_floor if not req.include_stale else 0.0,
        )
    except Exception as e:
        logger.warning("Keyword search failed: %s", e)

    # 4. Vector search
    vector_hits: list[tuple[str, float]] = []
    if have_embeddings and query_vec_bytes:
        try:
            vector_hits = storage.vector_search(
                query_vec_bytes, scope_ids,
                type_filter=types,
                include_stale=req.include_stale,
                limit=candidate_n,
                dimension=provider.dimension,
                value_floor=value_floor if not req.include_stale else 0.0,
            )
        except Exception as e:
            logger.warning("Vector search failed: %s", e)

    # 5. RRF fusion (preserves raw cosine/bm25 alongside the fused score)
    fused = _rrf_fuse(
        lists=[keyword_hits, vector_hits],
        list_names=["keyword", "vector"],
        k=k,
    )

    # 6. Filter by territory / tags if requested
    if req.territory or req.tags:
        fused = _post_filter(fused, storage, req.territory, req.tags)

    # 7. Hydrate. Iter 25 (Q-05): apply the per-fragment value multiplier
    # AFTER fusion. Iter 31: also apply a recency decay so old fragments
    # fade unless they're foundational (requirement/procedure/preference)
    # or pinned (value==1.0).
    frag_ids = [item[0] for item in fused]
    fragments_by_id = storage.get_fragments_by_ids(frag_ids)

    rescored: list[tuple[str, float, str, dict[str, float]]] = []
    for fid, rrf_score, matched_by, raw in fused:
        frag = fragments_by_id.get(fid)
        if frag is None:
            continue
        recency = _recency_multiplier(frag)
        adjusted = rrf_score * float(frag.value) * recency
        rescored.append((fid, adjusted, matched_by, raw))
    rescored.sort(key=lambda x: x[1], reverse=True)
    top = rescored[: req.limit]

    results: list[RecallResult] = []
    for rank, (fid, score, matched_by, raw) in enumerate(top, start=1):
        frag = fragments_by_id.get(fid)
        if frag is None:
            continue
        cosine = raw.get("vector")
        bm25_score = raw.get("keyword")
        results.append(RecallResult(
            fragment=frag, score=score, rank=rank, matched_by=matched_by,
            cosine=cosine, bm25=bm25_score,
            quality=classify_recall_quality(
                cosine=cosine, matched_by=matched_by, rank=rank,
            ),
        ))

    response = RecallResponse(
        results=results,
        query=req.query,
        scope=req.scope,
        total=len(results),
    )

    # 8. Iter 31: behavioural-value telemetry. Single batched UPDATE for
    # the top-K only, on a real (non-cached) response. Wrapped so it can
    # never break recall.
    if results:
        try:
            storage.bump_recall_hits([r.fragment.id for r in results])
        except Exception:
            logger.debug("recall_hits bump failed", exc_info=True)

    # 9. Iter 31: cache for 30 s
    _cache_set(req, response, scope_ids, storage_id)

    return response


# ---------------------------------------------------------------------------
# RRF implementation
# ---------------------------------------------------------------------------

def _rrf_fuse(
    lists: list[list[tuple[str, float]]],
    list_names: list[str],
    k: int = 60,
) -> list[tuple[str, float, str, dict[str, float]]]:
    """Fuse multiple ranked lists with RRF.

    Returns list of ``(id, rrf_score, source_name, raw_scores)`` sorted by
    score descending. ``raw_scores`` is a ``{list_name: original_score}`` map
    that lets callers surface the underlying signals (cosine, BM25) — the RRF
    score itself is just an ordinal-rank fusion artifact and is opaque to
    consumers without normalisation.
    """
    rrf_scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    raw_by_id: dict[str, dict[str, float]] = {}

    for ranked_list, name in zip(lists, list_names):
        for rank_0, (fid, raw_score) in enumerate(ranked_list):
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + 1.0 / (k + rank_0 + 1)
            sources.setdefault(fid, []).append(name)
            raw_by_id.setdefault(fid, {})[name] = float(raw_score)

    fused = []
    for fid, score in rrf_scores.items():
        src_list = sources[fid]
        matched_by = "hybrid" if len(src_list) > 1 else src_list[0]
        fused.append((fid, score, matched_by, raw_by_id[fid]))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def search_chunks(
    req: ChunkSearchRequest,
    storage: Storage,
    provider: EmbeddingProvider,
    *,
    k: int = 60,
    candidate_n: int = 30,
) -> ChunkSearchResponse:
    """Hybrid BM25 + vector + RRF over the chunks (codebase RAG) table.

    Same fusion as :func:`recall` but operates on the ``chunks`` table and
    accepts a ``ChunkSearchRequest`` (with optional language and root filters).
    """
    lineage = storage.get_scope_lineage(req.scope)
    if not lineage:
        return ChunkSearchResponse(
            results=[], query=req.query, scope=req.scope, total=0,
        )
    scope_ids = [s.id for s in lineage]
    languages = list(req.languages) if req.languages else None

    # Query embedding (best-effort)
    have_emb = False
    q_bytes: bytes = b""
    try:
        q_bytes = vec_to_bytes(provider.embed_one(req.query))
        have_emb = True
    except Exception as e:
        logger.warning("Chunk embedding failed, keyword-only: %s", e)

    # BM25
    keyword_hits: list[tuple[str, float]] = []
    try:
        keyword_hits = storage.chunks_keyword_search(
            req.query, scope_ids,
            languages=languages, source_root=req.source_root,
            limit=candidate_n,
        )
    except Exception as e:
        logger.warning("chunks keyword search failed: %s", e)

    # Vector
    vector_hits: list[tuple[str, float]] = []
    if have_emb:
        try:
            vector_hits = storage.chunks_vector_search(
                q_bytes, scope_ids,
                languages=languages, source_root=req.source_root,
                limit=candidate_n, dimension=provider.dimension,
            )
        except Exception as e:
            logger.warning("chunks vector search failed: %s", e)

    fused = _rrf_fuse(
        lists=[keyword_hits, vector_hits],
        list_names=["keyword", "vector"],
        k=k,
    )
    top = fused[: req.limit]
    chunk_ids = [item[0] for item in top]
    chunks_by_id = storage.get_chunks_by_ids(chunk_ids)

    results: list[ChunkSearchResult] = []
    for rank, (cid, score, matched_by, raw) in enumerate(top, start=1):
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        cosine = raw.get("vector")
        bm25_score = raw.get("keyword")
        results.append(ChunkSearchResult(
            chunk=chunk, score=score, rank=rank, matched_by=matched_by,
            cosine=cosine, bm25=bm25_score,
            quality=classify_recall_quality(
                cosine=cosine, matched_by=matched_by, rank=rank,
            ),
        ))

    return ChunkSearchResponse(
        results=results, query=req.query, scope=req.scope, total=len(results),
    )


def _post_filter(
    fused: list[tuple[str, float, str, dict[str, float]]],
    storage: Storage,
    territory: Optional[str],
    tags: Optional[list[str]],
) -> list[tuple[str, float, str, dict[str, float]]]:
    """Post-filter fused results by territory prefix and/or tags."""
    if not territory and not tags:
        return fused

    frag_ids = [item[0] for item in fused]
    frags = storage.get_fragments_by_ids(frag_ids)

    filtered = []
    for item in fused:
        fid = item[0]
        frag = frags.get(fid)
        if frag is None:
            continue
        if territory and (
            frag.territory is None
            or not frag.territory.startswith(territory)
        ):
            continue
        if tags and not any(t in frag.tags for t in tags):
            continue
        filtered.append(item)

    return filtered
