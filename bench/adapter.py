"""Adapter interface for context-bus tools under benchmark.

Two layers:

- ``ReadOnlyAdapter`` — recall, search, health. Safe against a live daemon.
- ``MutableAdapter`` — extends with write/ingest/reset. The standalone live
  reporter never calls these; the ephemeral pytest runner does.

Optional capabilities (typed-fragments, leases, OCC, git capture) are declared
via ``supports_*`` flags. Scenarios that require an unsupported capability
return ``ScenarioResult(status="skipped", reason=...)`` rather than failing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result shapes (tool-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class FragmentResult:
    """One fragment as returned by ``recall``."""

    id: str
    content: str
    type: str
    score: float = 0.0


@dataclass
class CodeChunkResult:
    """One code chunk as returned by ``search_code``."""

    id: str
    content: str
    file_path: str = ""
    score: float = 0.0


@dataclass
class HealthInfo:
    """What ``health()`` returns — tool-agnostic subset."""

    fragment_count: int = 0
    chunk_count: int = 0
    scope_count: int = 0
    version: str = ""
    tool: str = ""
    extra: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Read-only adapter — safe against a live daemon
# ---------------------------------------------------------------------------


class ReadOnlyAdapter(ABC):
    """Read-only operations. Implementations must not mutate state."""

    # Capability declarations — scenarios skip when False.
    supports_typed_fragments: bool = True
    supports_leases: bool = False
    supports_code_search: bool = True
    supports_scope_hierarchy: bool = False
    supports_git_capture: bool = False

    name: str = "unknown"

    @abstractmethod
    def health(self) -> HealthInfo:
        """Return tool health/stats."""

    @abstractmethod
    def recall(
        self,
        query: str,
        scope: str,
        *,
        limit: int = 10,
        types: list[str] | None = None,
    ) -> list[FragmentResult]:
        """Search the fragment bus by semantic+keyword. Ordered by score desc."""

    @abstractmethod
    def search_code(
        self,
        query: str,
        scope: str,
        *,
        limit: int = 10,
    ) -> list[CodeChunkResult]:
        """Search the indexed codebase. Ordered by score desc."""


# ---------------------------------------------------------------------------
# Mutable adapter — only used against ephemeral / test daemons
# ---------------------------------------------------------------------------


class MutableAdapter(ReadOnlyAdapter):
    """Adds write/ingest/reset. Never use against a live daemon."""

    @abstractmethod
    def reset(self) -> None:
        """Wipe all data. Destructive — only call against ephemeral DB."""

    @abstractmethod
    def ensure_scope(self, handle: str, *, parent: str | None = None) -> str:
        """Create scope if missing, return scope id/handle."""

    @abstractmethod
    def remember(
        self,
        content: str,
        *,
        type: str,
        scope: str,
        tags: list[str] | None = None,
        territory: str | None = None,
    ) -> str:
        """Store a fragment, return its id."""

    @abstractmethod
    def ingest_text(
        self,
        files: dict[str, str],
        *,
        scope: str,
        source_root: str = "bench",
    ) -> int:
        """Index a synthetic file map ({path: content}). Return chunk count."""

    # ---- Optional capabilities — default to NotImplementedError ----

    def capture_git_commits(self, repo_path: str, *, scope: str) -> int:
        """Process a git repo's commits into decision fragments."""
        raise NotImplementedError(f"{self.name}: git capture not supported")

    def would_capture_commit(self, subject: str, body: str = "") -> bool:
        """Algorithmic predicate: would this commit become a decision fragment?

        Used by the auto-capture-quality scenario. Default raises so tools
        without git capture skip the scenario rather than report bogus numbers.
        """
        raise NotImplementedError(f"{self.name}: git capture not supported")

    def claim_lease(self, glob: str, *, scope: str, ttl_seconds: int = 60) -> str | None:
        """Attempt to claim an advisory lease. Return id or None on conflict."""
        raise NotImplementedError(f"{self.name}: leases not supported")

    def release_lease(self, lease_id: str) -> None:
        raise NotImplementedError(f"{self.name}: leases not supported")
