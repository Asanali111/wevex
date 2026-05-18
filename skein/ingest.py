"""Codebase / document ingestion.

Walks a directory tree, splits each file into overlapping line-windows,
embeds each chunk, and stores the result in the `chunks` table.

Design notes:

- **Language-agnostic chunking.** v1 uses fixed-size line windows (default
  80 lines, 10-line overlap) regardless of language.  Symbol-aware splitting
  (functions, classes via tree-sitter) is a future enhancement; for retrieval
  quality at < 50k chunks the line-window approach with hybrid BM25+vector
  is competitive with symbol-aware approaches.

- **Incremental ingest.** Each chunk is keyed by (scope, root, path,
  line_start, line_end) and stamped with a sha256 of its content.  Re-running
  ``skein ingest`` against the same root reuses chunks whose hash hasn't
  changed (no re-embedding).  Files that disappeared between runs can be
  cleaned up with ``--prune``.

- **Embedding in batches.** We batch by ``EMBED_BATCH`` chunks per provider
  call to amortise API latency.  For the offline ``hash`` provider this
  doesn't matter; for Gemini / OpenAI it's a 5–50× speed-up.

- **No HTTP.** The CLI talks to ``Storage`` directly (same pattern as the
  hooks).  Daemon doesn't need to be running.
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .embeddings import EmbeddingProvider, vec_to_bytes
from .models import ChunkCreate
from .storage import Storage

logger = logging.getLogger("skein.ingest")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

EMBED_BATCH = 32                   # chunks per embedding-provider call

# File-extension → language label.  Add more as needed.
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hxx": "cpp",
    ".cs": "csharp",
    ".scala": "scala",
    ".clj": "clojure", ".cljs": "clojure",
    ".ex": "elixir", ".exs": "elixir",
    ".elm": "elm",
    ".hs": "haskell",
    ".lua": "lua",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".sql": "sql",
    ".md": "markdown", ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".json": "json", ".jsonc": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".vue": "vue", ".svelte": "svelte",
    ".dockerfile": "dockerfile",
    ".tf": "terraform",
    ".proto": "protobuf",
    ".graphql": "graphql", ".gql": "graphql",
    ".r": "r",
    ".dart": "dart",
}

# Files / dirs we never ingest by default.
# Anything that's user-private state, agent-cache, build output, or third-party
# package source. Privacy categories are flagged so a future audit can verify
# we never index browser/credential data.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    # VCS
    ".git", ".hg", ".svn",
    # Language / build caches
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".cache",
    "venv", ".venv", "env",
    "dist", "build", "target",
    ".next", ".nuxt", ".turbo", ".vercel",
    "coverage", "htmlcov",
    ".idea", ".vscode",
    "_archive_v2", "_archive",
    # macOS / OS user-state (never ingest if someone runs from $HOME)
    "Library", "Applications", "Music", "Movies", "Pictures",
    "Public", ".Trash", ".DocumentRevisions-V100", ".Spotlight-V100",
    ".fseventsd", ".TemporaryItems",
    # Agent/AI tool caches and credentials
    ".claude", ".cursor", ".gemini", ".codex", ".antigravity",
    ".skein", ".aider.cache",
    # Local-state for tools we don't want to read
    ".local", ".config",  # only matters when ingesting from $HOME
    ".npm", ".yarn", ".pnpm-store",
    ".rustup", ".cargo", ".rbenv", ".nvm", ".pyenv",
    ".docker", ".kube", ".aws", ".gcloud", ".azure",
    ".ssh", ".gnupg", ".password-store",
    # Browser profiles / password databases (privacy)
    "Chrome", "Firefox", "Safari", "Edge", "Brave", "Arc",
    "ZxcvbnData",
)


# Filename substrings that flag a file as private and skip-on-sight,
# even if it has an extension we'd otherwise ingest.
SENSITIVE_FILENAME_FRAGMENTS: tuple[str, ...] = (
    "passwords", "credential", "id_rsa", "id_ed25519", "id_ecdsa",
    ".env", "secret", "api_key", "apikey",
)


# Path roots Skein refuses to ingest from — a project directory should never
# match any of these. Trying to run `skein up` here exits with an error.
_SYSTEM_ROOTS = {
    "/", "/Users", "/home", "/tmp", "/var", "/etc",
    # macOS resolves /tmp → /private/tmp, /var → /private/var
    "/private", "/private/tmp", "/private/var", "/private/etc",
    "/System", "/Library", "/Applications",
}


def _refuse_root(root: Path) -> Optional[str]:
    """Return a reason string if ``root`` is a forbidden ingest target."""
    raw = str(root)
    resolved = root.resolve()
    rs = str(resolved)
    home = str(Path.home().resolve())

    # Root or system dirs (check both raw input and resolved target —
    # /tmp on macOS resolves to /private/tmp)
    if raw in _SYSTEM_ROOTS or rs in _SYSTEM_ROOTS:
        return f"system directory ({raw})"

    # The user's own $HOME — auto-detect would call this 'project:<username>'
    if rs == home:
        return f"$HOME ({rs}) — pick a real project directory inside it"

    # macOS top-level user dirs that are not project material
    if resolved.parent == Path(home) and resolved.name in {
        "Library", "Applications", "Music", "Movies", "Pictures",
        "Public", "Desktop", "Downloads",
    }:
        return f"system folder under $HOME ({resolved.name})"

    return None

# Hard caps to protect the system
MAX_FILE_BYTES = 512 * 1024        # 512 KB per file
MAX_CHUNK_CHARS = 8000             # truncate any single chunk
DEFAULT_INCLUDE_EXTS = frozenset(LANGUAGE_BY_EXT.keys())


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class IngestStats:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    chunks_inserted: int = 0
    chunks_unchanged: int = 0
    chunks_updated: int = 0
    chunks_pruned: int = 0
    bytes_processed: int = 0
    skipped_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    embedding_degraded: bool = False  # set if the provider gave up mid-run


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_directory(
    root: Path,
    storage: Storage,
    provider: Optional[EmbeddingProvider],
    *,
    scope_id: str,
    source_root: str,
    chunk_lines: int = 80,
    overlap_lines: int = 10,
    include_exts: Optional[Iterable[str]] = None,
    extra_excludes: Iterable[str] = (),
    max_file_bytes: int = MAX_FILE_BYTES,
    prune_missing: bool = False,
    dry_run: bool = False,
    progress_cb: Optional[Callable[[str, IngestStats], None]] = None,
) -> IngestStats:
    """Ingest every supported file under ``root`` into the chunks table.

    Parameters
    ----------
    root:
        Directory to walk.
    storage:
        Open Storage instance.
    provider:
        Embedding provider, or None to skip embeddings (keyword search only).
    scope_id:
        Scope UUID to attribute chunks to.
    source_root:
        Stable label used in the DB (typically the basename of ``root``).
    chunk_lines, overlap_lines:
        Line-window sizing.
    include_exts:
        Iterable of ``.ext`` strings.  Defaults to all known languages.
    extra_excludes:
        Additional directory/file glob patterns to skip.
    max_file_bytes:
        Files larger than this are skipped.
    prune_missing:
        Delete chunks under ``source_root`` whose source file is no longer
        present.  Compares against (root, source_path) pairs seen this run.
    dry_run:
        Walk and chunk but do not write anything.
    progress_cb:
        Called as ``cb(relative_path, stats)`` after each file.
    """
    stats = IngestStats()
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    refusal = _refuse_root(root)
    if refusal:
        raise ValueError(
            f"refusing to ingest from {refusal}. "
            "Pass an explicit project directory."
        )

    include_set: set[str] = (
        set(include_exts) if include_exts is not None else set(DEFAULT_INCLUDE_EXTS)
    )
    excludes = set(DEFAULT_EXCLUDES) | set(extra_excludes)

    seen_paths: set[tuple[str, str]] = set()  # (source_root, source_path)
    pending: list[tuple[ChunkCreate, str]] = []   # (chunk, content_hash)

    for path in _walk(root, include_set, excludes):
        rel = path.relative_to(root).as_posix()
        stats.files_seen += 1

        # Sensitive-filename filter — never ingest passwords/credentials etc.
        lc = path.name.lower()
        if any(frag in lc for frag in SENSITIVE_FILENAME_FRAGMENTS):
            stats.files_skipped += 1
            stats.skipped_paths.append(f"{rel} (sensitive filename)")
            continue

        try:
            size = path.stat().st_size
        except OSError as e:
            stats.errors.append(f"{rel}: {e}")
            continue

        if size > max_file_bytes:
            stats.files_skipped += 1
            stats.skipped_paths.append(f"{rel} ({size} > {max_file_bytes})")
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            stats.files_skipped += 1
            stats.skipped_paths.append(f"{rel} (binary / non-utf8)")
            continue
        except OSError as e:
            stats.errors.append(f"{rel}: {e}")
            continue

        stats.bytes_processed += size
        seen_paths.add((source_root, rel))

        language = LANGUAGE_BY_EXT.get(path.suffix.lower())
        chunks = _chunk_text(text, chunk_lines=chunk_lines, overlap=overlap_lines)
        if not chunks:
            continue
        stats.files_ingested += 1

        for c in chunks:
            content = c["content"][:MAX_CHUNK_CHARS]
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk_create = ChunkCreate(
                scope_id=scope_id,
                source_root=source_root,
                source_path=rel,
                content=content,
                line_start=c["line_start"],
                line_end=c["line_end"],
                language=language,
                chunk_type="window",
                metadata={"size_bytes": len(content)},
            )
            pending.append((chunk_create, content_hash))

            if len(pending) >= EMBED_BATCH:
                _flush_batch(pending, storage, provider, stats, dry_run=dry_run)
                pending = []

        if progress_cb:
            progress_cb(rel, stats)

    if pending:
        _flush_batch(pending, storage, provider, stats, dry_run=dry_run)

    if prune_missing and not dry_run:
        stats.chunks_pruned = _prune_missing(
            storage, scope_id, source_root, seen_paths,
        )

    return stats


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def count_ingestable_files(
    root: Path,
    *,
    include_exts: Optional[Iterable[str]] = None,
    extra_excludes: Iterable[str] = (),
) -> int:
    """Pre-walk count for progress bars + safety prompts. Cheap (no read)."""
    root = root.resolve()
    include_set: set[str] = (
        set(include_exts) if include_exts is not None else set(DEFAULT_INCLUDE_EXTS)
    )
    excludes = set(DEFAULT_EXCLUDES) | set(extra_excludes)
    return sum(1 for _ in _walk(root, include_set, excludes))


def _walk(
    root: Path,
    include_exts: set[str],
    excludes: set[str],
) -> Iterable[Path]:
    """Yield every file under ``root`` whose extension is in ``include_exts``,
    skipping any directory whose name matches an entry in ``excludes`` and any
    file path matching an exclude glob."""
    glob_excludes = [p for p in excludes if any(c in p for c in "*?[")]
    name_excludes = {p for p in excludes if not any(c in p for c in "*?[")}

    for dirpath, dirnames, filenames in os.walk(root):
        # Mutate dirnames in-place to prune entire subtrees
        dirnames[:] = [d for d in dirnames if d not in name_excludes]
        for fname in filenames:
            path = Path(dirpath) / fname
            ext = path.suffix.lower()
            if ext not in include_exts:
                continue
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, g) for g in glob_excludes):
                continue
            if any(part in name_excludes for part in path.parts):
                continue
            yield path


def _chunk_text(
    text: str, *, chunk_lines: int = 80, overlap: int = 10,
) -> list[dict[str, Any]]:
    """Split ``text`` into overlapping line-windows.

    Returns a list of dicts: ``{"content": str, "line_start": 1-based,
    "line_end": 1-based inclusive}``.
    """
    if not text.strip():
        return []
    if chunk_lines < 1:
        raise ValueError("chunk_lines must be >= 1")
    if overlap < 0 or overlap >= chunk_lines:
        overlap = max(0, min(overlap, chunk_lines - 1))

    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    step = max(1, chunk_lines - overlap)
    out: list[dict[str, Any]] = []
    i = 0
    while i < n:
        j = min(n, i + chunk_lines)
        slice_text = "\n".join(lines[i:j]).strip()
        if slice_text:
            out.append({
                "content": "\n".join(lines[i:j]),
                "line_start": i + 1,
                "line_end": j,
            })
        if j == n:
            break
        i += step
    return out


def _flush_batch(
    pending: list[tuple[ChunkCreate, str]],
    storage: Storage,
    provider: Optional[EmbeddingProvider],
    stats: IngestStats,
    *,
    dry_run: bool,
) -> None:
    """Bulk-aware flush.

    Old behaviour: embed every pending chunk, then upsert one-by-one. For a
    re-ingest where nothing changed (the common case) this still paid the
    full per-chunk embedding cost. On the user's machine that meant 264
    chunks × 1 Gemini batch call each restart, even though zero rows would
    be written.

    New behaviour:
      1. Bulk-fetch existing ``content_hash`` for every key in this batch.
      2. Partition into ``unchanged`` (hash matches → skip both embed and
         upsert) and ``changed`` (new or hash-mismatch → must embed + upsert).
      3. Embed only the changed subset.
      4. Wrap the upserts in a single transaction so we pay one fsync per
         batch instead of one per chunk.
    """
    if dry_run:
        stats.chunks_inserted += len(pending)
        return

    if not pending:
        return

    # ---- 1. Bulk lookup of existing hashes for this batch ----
    # All chunks in the same batch share scope_id + source_root (they may
    # span multiple files within the same ingest). Look them up in one query.
    scope_id = pending[0][0].scope_id
    source_root = pending[0][0].source_root
    keys = [
        (p[0].source_path, p[0].line_start, p[0].line_end) for p in pending
    ]

    bulk = getattr(storage, "bulk_get_chunk_hashes", None)
    if callable(bulk):
        existing_hash_by_key = bulk(scope_id, source_root, keys)
    else:
        existing_hash_by_key = {}

    # ---- 2. Partition ----
    changed_indices: list[int] = []
    for i, (chunk_create, content_hash) in enumerate(pending):
        key = (chunk_create.source_path,
               chunk_create.line_start, chunk_create.line_end)
        if existing_hash_by_key.get(key) != content_hash:
            changed_indices.append(i)
    n_unchanged = len(pending) - len(changed_indices)
    stats.chunks_unchanged += n_unchanged

    # ---- 3. Embed only the changed subset ----
    embeddings_by_idx: dict[int, bytes] = {}
    if changed_indices and provider is not None:
        try:
            texts = [pending[i][0].content for i in changed_indices]
            vecs = provider.embed(texts)
            for idx, vec in zip(changed_indices, vecs):
                embeddings_by_idx[idx] = vec_to_bytes(vec)
            if getattr(provider, "degraded", False):
                stats.embedding_degraded = True
        except Exception as e:
            logger.warning("embedding batch failed; skipping vectors: %s", e)
            stats.errors.append(f"embedding failed: {e}")

    if not changed_indices:
        # Nothing to write — return now and skip the txn entirely.
        return

    # ---- 4. Upsert only the changed subset in a single transaction ----
    # Test fakes that don't speak SQL skip the explicit transaction.
    begin = getattr(storage, "begin_immediate", None)
    commit = getattr(storage, "commit_immediate", None)
    rollback = getattr(storage, "rollback_immediate", None)
    in_txn = callable(begin) and callable(commit)
    if in_txn:
        begin()
    try:
        for idx in changed_indices:
            chunk_create, content_hash = pending[idx]
            emb = embeddings_by_idx.get(idx)
            try:
                _, status = storage.upsert_chunk(
                    chunk_create, content_hash=content_hash, embedding=emb,
                )
                if status == "inserted":
                    stats.chunks_inserted += 1
                elif status == "updated":
                    stats.chunks_updated += 1
                else:  # rare: race won by another writer
                    stats.chunks_unchanged += 1
            except Exception as e:
                stats.errors.append(
                    f"{chunk_create.source_path}:{chunk_create.line_start}: {e}"
                )
        if in_txn:
            commit()
    except Exception:
        if in_txn and callable(rollback):
            try:
                rollback()
            except Exception:
                pass
        raise


def _prune_missing(
    storage: Storage,
    scope_id: str,
    source_root: str,
    seen: set[tuple[str, str]],
) -> int:
    """Delete chunks whose (source_root, source_path) wasn't seen this run.

    Only ``DISTINCT source_path`` is pulled into Python — not every chunk row,
    which on a large project could be millions. Deletes are batched into
    ``IN (?, ?, …)`` groups to amortise the per-statement fsync cost.
    """
    seen_paths = {p for _, p in seen}
    rows = storage._conn.execute(
        "SELECT DISTINCT source_path FROM chunks "
        "WHERE scope_id = ? AND source_root = ?",
        (scope_id, source_root),
    ).fetchall()
    to_delete = [r[0] for r in rows if r[0] not in seen_paths]
    if not to_delete:
        return 0
    BATCH = 500
    n = 0
    for i in range(0, len(to_delete), BATCH):
        batch = to_delete[i:i + BATCH]
        placeholders = ",".join("?" * len(batch))
        n += storage._conn.execute(
            f"DELETE FROM chunks WHERE scope_id = ? AND source_root = ? "
            f"AND source_path IN ({placeholders})",
            [scope_id, source_root, *batch],
        ).rowcount
    return n
