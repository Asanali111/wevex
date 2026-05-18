"""Pydantic models for Skein's domain objects.

All IDs are UUID strings. Timestamps are ISO-8601 UTC strings. Enums are
validated by both Pydantic and the DB CHECK constraints.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums (kept as literal string sets to stay compatible with Python 3.9)
# ---------------------------------------------------------------------------

IDENTITY_TYPES = frozenset({"user", "agent", "llm", "service"})
SCOPE_TYPES = frozenset({"public", "org", "team", "project", "personal"})
MEMBERSHIP_ROLES = frozenset({"owner", "admin", "contributor", "viewer"})
FRAGMENT_TYPES = frozenset({
    "preference", "fact", "decision", "state",
    "observation", "requirement", "procedure", "conversation",
})

# Default TTL by fragment type (seconds).  None = permanent.
FRAGMENT_DEFAULT_TTL: dict[str, Optional[int]] = {
    "preference":   90 * 86400,
    "fact":         30 * 86400,
    "decision":     30 * 86400,
    "state":         7 * 86400,
    "observation":  14 * 86400,
    "requirement":  None,
    "procedure":    None,
    "conversation": 30 * 86400,
}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class IdentityCreate(BaseModel):
    handle: str = Field(..., description="Unique namespaced handle, e.g. 'user:ameliomar'")
    type: str = Field(..., description=f"One of {sorted(IDENTITY_TYPES)}")
    name: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in IDENTITY_TYPES:
            raise ValueError(f"type must be one of {sorted(IDENTITY_TYPES)}")
        return v


class Identity(IdentityCreate):
    id: str = Field(default_factory=_new_id)
    created_at: str = Field(default_factory=_now_iso)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

class ScopeCreate(BaseModel):
    handle: str = Field(..., description="Unique scope handle, e.g. 'project:skein'")
    type: str = Field(..., description=f"One of {sorted(SCOPE_TYPES)}")
    name: str
    parent_scope_id: Optional[str] = None
    owner_id: str

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in SCOPE_TYPES:
            raise ValueError(f"type must be one of {sorted(SCOPE_TYPES)}")
        return v


class Scope(ScopeCreate):
    id: str = Field(default_factory=_new_id)
    created_at: str = Field(default_factory=_now_iso)

    model_config = ConfigDict(from_attributes=True)


class ScopeMembershipCreate(BaseModel):
    scope_id: str
    identity_id: str
    role: str = "contributor"

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in MEMBERSHIP_ROLES:
            raise ValueError(f"role must be one of {sorted(MEMBERSHIP_ROLES)}")
        return v


class ScopeMembership(ScopeMembershipCreate):
    id: str = Field(default_factory=_new_id)
    granted_at: str = Field(default_factory=_now_iso)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------

class CommitCreate(BaseModel):
    author_id: str
    scope_id: str
    message: str
    parent_commit_id: Optional[str] = None
    fragments_added: list[str] = Field(default_factory=list)
    fragments_modified: list[str] = Field(default_factory=list)
    fragments_removed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Commit(CommitCreate):
    id: str = Field(default_factory=_new_id)
    created_at: str = Field(default_factory=_now_iso)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------

class FragmentCreate(BaseModel):
    type: str = Field(..., description=f"One of {sorted(FRAGMENT_TYPES)}")
    content: str = Field(..., min_length=1)
    scope_id: str
    owner_id: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    ttl_seconds: Optional[int] = Field(None, description="Override default TTL. 0 = permanent.")
    tags: list[str] = Field(default_factory=list)
    territory: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Provenance (iter 14.0). All optional — explicit MCP calls fill these
    # automatically; manual REST calls may leave them blank.
    created_by_tool: Optional[str] = None
    created_in_session_id: Optional[str] = None
    created_against_commit: Optional[str] = None
    files_open_at_creation: list[str] = Field(default_factory=list)
    supersedes_fragment_id: Optional[str] = None
    extraction_method: str = Field("explicit", description="explicit | code-scan | transcript-claude | …")
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in FRAGMENT_TYPES:
            raise ValueError(f"type must be one of {sorted(FRAGMENT_TYPES)}")
        return v

    @model_validator(mode="after")
    def _set_ttl_default(self) -> "FragmentCreate":
        """Apply type-based default TTL unless caller set ttl_seconds explicitly."""
        if self.ttl_seconds is None:
            default = FRAGMENT_DEFAULT_TTL.get(self.type)
            if default is not None:
                self.ttl_seconds = default
            # if default is None → permanent
        elif self.ttl_seconds == 0:
            self.ttl_seconds = None  # 0 means permanent
        return self


class FragmentUpdate(BaseModel):
    """All fields optional — PATCH semantics."""
    content: Optional[str] = Field(None, min_length=1)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[list[str]] = None
    territory: Optional[str] = None
    is_stale: Optional[bool] = None
    stale_reason: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    # For OCC: caller must send the version they last read.
    expected_version: int = Field(..., description="Optimistic-concurrency version from last read")


class Fragment(FragmentCreate):
    id: str = Field(default_factory=_new_id)
    version: int = 1
    permanent: bool = False
    expires_at: Optional[str] = None
    is_stale: bool = False
    stale_reason: Optional[str] = None
    source_commit_id: Optional[str] = None
    superseded_by_fragment_id: Optional[str] = None  # mirror of the FK in DB
    # Iter 25 (Q-05): deterministic "is this worth recalling?" score in
    # [0.05, 1.0]. Set at create_fragment from provenance + type + content;
    # multiplied into the final RRF score in retrieval.recall.
    value: float = 0.5
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Lease
# ---------------------------------------------------------------------------

class LeaseCreate(BaseModel):
    scope_id: str
    glob: str = Field(..., description="File-glob pattern, e.g. 'backend/auth/**'")
    owner_id: str
    ttl_seconds: int = Field(300, description="How long to hold the lease (seconds). Default 5 min.")
    reason: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Lease(LeaseCreate):
    id: str = Field(default_factory=_new_id)
    acquired_at: str = Field(default_factory=_now_iso)
    expires_at: str = ""  # set by storage layer

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Chunks (codebase / document RAG)
# ---------------------------------------------------------------------------

CHUNK_TYPES = frozenset({"window", "section", "file", "symbol"})


class ChunkCreate(BaseModel):
    """A slice of source code or document content, indexed for RAG."""
    scope_id: str
    source_root: str = Field(..., description="Stable label for the ingest base (project dir).")
    source_path: str = Field(..., description="Path relative to source_root (forward slashes).")
    content: str = Field(..., min_length=1)
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)
    language: Optional[str] = None
    chunk_type: str = "window"
    symbol_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_type")
    @classmethod
    def _validate_chunk_type(cls, v: str) -> str:
        if v not in CHUNK_TYPES:
            raise ValueError(f"chunk_type must be one of {sorted(CHUNK_TYPES)}")
        return v


class Chunk(ChunkCreate):
    id: str = Field(default_factory=_new_id)
    content_hash: str = ""
    created_at: str = Field(default_factory=_now_iso)

    model_config = ConfigDict(from_attributes=True)


class ChunkSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    scope: str
    languages: Optional[list[str]] = None
    source_root: Optional[str] = None
    limit: int = Field(10, ge=1, le=50)


class ChunkSearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    matched_by: str
    cosine: Optional[float] = None
    bm25: Optional[float] = None
    quality: str = "none"


class ChunkSearchResponse(BaseModel):
    results: list[ChunkSearchResult]
    query: str
    scope: str
    total: int


class ChunkStats(BaseModel):
    scope: str
    total_chunks: int
    total_files: int
    by_language: dict[str, int]
    by_root: dict[str, int]


# ---------------------------------------------------------------------------
# Search / recall
# ---------------------------------------------------------------------------

class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1)
    scope: str
    types: Optional[list[str]] = None
    territory: Optional[str] = None
    tags: Optional[list[str]] = None
    limit: int = Field(10, ge=1, le=50)
    include_stale: bool = False

    @field_validator("types", mode="before")
    @classmethod
    def _validate_types(cls, v):
        if v is not None:
            bad = [t for t in v if t not in FRAGMENT_TYPES]
            if bad:
                raise ValueError(f"Unknown fragment types: {bad}")
        return v


RECALL_QUALITY_LEVELS = ("high", "medium", "low", "none")

# Cosine thresholds calibrated for normalized 384-dim sentence embeddings
# (BAAI/bge-small-en-v1.5, the iter-23 default). Lower bounds for each bucket.
# Hand-set so that "high" requires real paraphrase agreement, "none" is the
# honest no-knowledge signal that the AGENTS.md guidance was always meant to
# trigger (the old RRF-score threshold could not — RRF caps at ~0.033).
RECALL_QUALITY_THRESHOLDS = {
    "high":   0.65,
    "medium": 0.50,
    "low":    0.35,
}


def classify_recall_quality(
    *,
    cosine: Optional[float],
    matched_by: str,
    rank: int,
) -> str:
    """Map a result's raw signals to a human-readable quality bucket.

    Cosine is authoritative when available — a high vector similarity means
    the query and fragment are semantically close, which is what callers care
    about. Pure-keyword hits (no embedding overlap) earn at most "low" because
    BM25 by itself often spuriously matches stopwords or boilerplate.
    """
    if cosine is not None:
        if cosine >= RECALL_QUALITY_THRESHOLDS["high"]:
            return "high"
        if cosine >= RECALL_QUALITY_THRESHOLDS["medium"]:
            return "medium"
        if cosine >= RECALL_QUALITY_THRESHOLDS["low"]:
            return "low"
        return "none"
    # Keyword-only fallback: a top-3 BM25 hit is plausibly relevant, but the
    # caller should know there's no semantic signal backing it.
    if matched_by == "keyword" and rank <= 3:
        return "low"
    return "none"


class RecallResult(BaseModel):
    fragment: Fragment
    score: float
    rank: int
    matched_by: str  # "keyword", "vector", "hybrid"
    # iter 24: expose raw signals so callers don't have to interpret the
    # opaque RRF score. ``cosine`` is the underlying vector similarity in
    # [-1, 1] when a vector hit contributed; ``bm25`` is the (negated) BM25
    # relevance when a keyword hit contributed. ``quality`` is the
    # human-readable bucket — prefer it over ``score`` for routing decisions.
    cosine: Optional[float] = None
    bm25: Optional[float] = None
    quality: str = "none"


class RecallResponse(BaseModel):
    results: list[RecallResult]
    query: str
    scope: str
    total: int


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    db_path: str = ""
    fragment_count: int = 0
    scope_count: int = 0
    identity_count: int = 0


# ---------------------------------------------------------------------------
# Error envelopes
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
