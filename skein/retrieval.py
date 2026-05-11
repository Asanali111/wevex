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
from typing import Dict, List, Optional, Tuple

from .embeddings import EmbeddingProvider, vec_to_bytes
from .models import (
    ChunkSearchRequest, ChunkSearchResponse, ChunkSearchResult,
    Fragment, RecallRequest, RecallResponse, RecallResult,
)
from .storage import Storage

logger = logging.getLogger("skein.retrieval")


def recall(
    req: RecallRequest,
    storage: Storage,
    provider: EmbeddingProvider,
    *,
    k: int = 60,            # RRF constant
    candidate_n: int = 30,  # candidates per list before fusion
) -> RecallResponse:
    """Main entry point for hybrid recall.

    Steps:
    1. Resolve scope lineage (query scope + all ancestors).
    2. Embed the query.
    3. BM25 keyword search → ranked list A.
    4. Vector cosine search → ranked list B.
    5. Fuse A + B with RRF.
    6. Hydrate top-N fragments.
    7. Return RecallResponse.
    """
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
    keyword_hits: List[Tuple[str, float]] = []
    try:
        keyword_hits = storage.keyword_search(
            req.query, scope_ids,
            type_filter=types,
            include_stale=req.include_stale,
            limit=candidate_n,
        )
    except Exception as e:
        logger.warning("Keyword search failed: %s", e)

    # 4. Vector search
    vector_hits: List[Tuple[str, float]] = []
    if have_embeddings and query_vec_bytes:
        try:
            vector_hits = storage.vector_search(
                query_vec_bytes, scope_ids,
                type_filter=types,
                include_stale=req.include_stale,
                limit=candidate_n,
                dimension=provider.dimension,
            )
        except Exception as e:
            logger.warning("Vector search failed: %s", e)

    # 5. RRF fusion
    fused = _rrf_fuse(
        lists=[keyword_hits, vector_hits],
        list_names=["keyword", "vector"],
        k=k,
    )

    # 6. Filter by territory / tags if requested
    if req.territory or req.tags:
        fused = _post_filter(fused, storage, req.territory, req.tags)

    # 7. Hydrate
    top = fused[: req.limit]
    frag_ids = [fid for fid, _, _ in top]
    fragments_by_id = storage.get_fragments_by_ids(frag_ids)

    results: List[RecallResult] = []
    for rank, (fid, score, matched_by) in enumerate(top, start=1):
        frag = fragments_by_id.get(fid)
        if frag is None:
            continue
        results.append(RecallResult(
            fragment=frag, score=score, rank=rank, matched_by=matched_by,
        ))

    return RecallResponse(
        results=results,
        query=req.query,
        scope=req.scope,
        total=len(results),
    )


# ---------------------------------------------------------------------------
# RRF implementation
# ---------------------------------------------------------------------------

def _rrf_fuse(
    lists: List[List[Tuple[str, float]]],
    list_names: List[str],
    k: int = 60,
) -> List[Tuple[str, float, str]]:
    """Fuse multiple ranked lists with RRF.

    Returns list of (fragment_id, rrf_score, source_name) sorted desc by score.
    source_name indicates which list(s) contributed (e.g. "hybrid").
    """
    rrf_scores: Dict[str, float] = {}
    sources: Dict[str, List[str]] = {}

    for ranked_list, name in zip(lists, list_names):
        for rank_0, (fid, _raw_score) in enumerate(ranked_list):
            rrf_scores[fid] = rrf_scores.get(fid, 0.0) + 1.0 / (k + rank_0 + 1)
            sources.setdefault(fid, []).append(name)

    fused = []
    for fid, score in rrf_scores.items():
        src_list = sources[fid]
        matched_by = "hybrid" if len(src_list) > 1 else src_list[0]
        fused.append((fid, score, matched_by))

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
    keyword_hits: List[Tuple[str, float]] = []
    try:
        keyword_hits = storage.chunks_keyword_search(
            req.query, scope_ids,
            languages=languages, source_root=req.source_root,
            limit=candidate_n,
        )
    except Exception as e:
        logger.warning("chunks keyword search failed: %s", e)

    # Vector
    vector_hits: List[Tuple[str, float]] = []
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
    chunk_ids = [cid for cid, _, _ in top]
    chunks_by_id = storage.get_chunks_by_ids(chunk_ids)

    results: List[ChunkSearchResult] = []
    for rank, (cid, score, matched_by) in enumerate(top, start=1):
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        results.append(ChunkSearchResult(
            chunk=chunk, score=score, rank=rank, matched_by=matched_by,
        ))

    return ChunkSearchResponse(
        results=results, query=req.query, scope=req.scope, total=len(results),
    )


def _post_filter(
    fused: List[Tuple[str, float, str]],
    storage: Storage,
    territory: Optional[str],
    tags: Optional[List[str]],
) -> List[Tuple[str, float, str]]:
    """Post-filter fused results by territory prefix and/or tags."""
    if not territory and not tags:
        return fused

    frag_ids = [fid for fid, _, _ in fused]
    frags = storage.get_fragments_by_ids(frag_ids)

    filtered = []
    for item in fused:
        fid, score, src = item
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
