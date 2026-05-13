"""Filesystem watcher: incremental re-ingest when files change.

Two backends:
  • **watchdog** (preferred) — native FSEvents on macOS, inotify on Linux.
    Sub-second detection, low CPU.
  • **polling fallback** — pure-Python, scans mtimes every ``poll_interval``
    seconds. Works without any extra dependency.

The watcher is started by the daemon (one per registered project) and runs
inside the FastAPI lifespan task group.  When a file under the project root
changes:

    1. Skip if it's outside the include set or in the exclude set.
    2. Debounce ``debounce_secs`` to coalesce rapid saves.
    3. Re-chunk + re-embed *just that file* and upsert into ``chunks``.

Deletions are handled too — when a watched file disappears, its chunks are
removed.

The watcher talks to the same ``Storage`` instance the rest of the daemon
uses, so writes flow through SQLite WAL and are immediately visible to
``search_chunks``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .embeddings import EmbeddingProvider, vec_to_bytes
from .ingest import (
    DEFAULT_EXCLUDES,
    DEFAULT_INCLUDE_EXTS,
    LANGUAGE_BY_EXT,
    MAX_CHUNK_CHARS,
    MAX_FILE_BYTES,
    _chunk_text,
)
from .models import ChunkCreate
from .storage import Storage

logger = logging.getLogger("skein.watcher")

# How long to wait after the last change before re-ingesting (coalesce saves)
DEFAULT_DEBOUNCE = 1.5

# How often the polling backend rescans
DEFAULT_POLL_INTERVAL = 3.0


@dataclass
class WatchStats:
    files_reingested: int = 0
    files_deleted: int = 0
    errors: int = 0


class _BaseWatcher:
    """Shared logic — both backends call ``_handle_change``/``_handle_delete``."""

    def __init__(
        self,
        root: Path,
        scope_id: str,
        source_root: str,
        storage: Storage,
        provider: EmbeddingProvider | None,
        *,
        chunk_lines: int = 80,
        overlap_lines: int = 10,
        include_exts: Iterable[str] | None = None,
        excludes: Iterable[str] = (),
        debounce_secs: float = DEFAULT_DEBOUNCE,
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        self.root = root.resolve()
        self.scope_id = scope_id
        self.source_root = source_root
        self.storage = storage
        self.provider = provider
        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines
        self.include_set: set[str] = (
            set(include_exts) if include_exts is not None else set(DEFAULT_INCLUDE_EXTS)
        )
        self.excludes = set(DEFAULT_EXCLUDES) | set(excludes)
        self.debounce_secs = debounce_secs
        self.max_file_bytes = max_file_bytes
        self.stats = WatchStats()

        # Pending re-ingests, keyed by absolute path.  Value is the time the
        # file last changed; we re-ingest when (now - pending[path]) >= debounce.
        self._pending: dict[Path, float] = {}
        self._pending_lock = threading.Lock()

        self._stop_event = threading.Event()

    # ----- public surface -----------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()

    # ----- change dispatch ----------------------------------------------

    def _should_track(self, path: Path) -> bool:
        if not path.is_absolute():
            return False
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return False
        # Path component excludes
        if any(part in self.excludes for part in rel.parts):
            return False
        # Extension include set
        return path.suffix.lower() in self.include_set

    def _enqueue(self, path: Path) -> None:
        if not self._should_track(path):
            return
        with self._pending_lock:
            self._pending[path] = time.time()

    def _enqueue_deleted(self, path: Path) -> None:
        # Even if the file's gone, we want to remove its chunks.
        try:
            rel = path.relative_to(self.root).as_posix()
        except ValueError:
            return
        try:
            n = self.storage._conn.execute(
                "DELETE FROM chunks WHERE scope_id = ? AND source_root = ? AND source_path = ?",
                (self.scope_id, self.source_root, rel),
            ).rowcount
            if n:
                self.stats.files_deleted += 1
                logger.info("watcher: removed %d chunks for deleted %s", n, rel)
        except Exception as e:
            logger.warning("watcher: delete failed for %s: %s", rel, e)
            self.stats.errors += 1

    def _drain_pending(self) -> None:
        """Re-ingest any pending paths whose debounce window has elapsed."""
        now = time.time()
        ready: list[Path] = []
        with self._pending_lock:
            for path, ts in list(self._pending.items()):
                if now - ts >= self.debounce_secs:
                    ready.append(path)
                    del self._pending[path]
        for path in ready:
            self._reingest(path)

    def _reingest(self, path: Path) -> None:
        if not path.exists():
            self._enqueue_deleted(path)
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size > self.max_file_bytes:
            return
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return

        rel = path.relative_to(self.root).as_posix()
        language = LANGUAGE_BY_EXT.get(path.suffix.lower())
        chunks = _chunk_text(
            text, chunk_lines=self.chunk_lines, overlap=self.overlap_lines,
        )
        if not chunks:
            return

        # Embed the batch
        contents = [c["content"][:MAX_CHUNK_CHARS] for c in chunks]
        embeddings: list[bytes | None] = [None] * len(contents)
        if self.provider is not None:
            try:
                vecs = self.provider.embed(contents)
                embeddings = [vec_to_bytes(v) for v in vecs]
            except Exception as e:
                logger.warning("watcher: embedding failed for %s: %s", rel, e)

        # Single transaction for the whole file save: N upserts + a stale-prune
        # would otherwise be N+1 fsyncs. Live editing fires this on every save,
        # so the savings compound.
        begin = getattr(self.storage, "begin_immediate", None)
        commit = getattr(self.storage, "commit_immediate", None)
        rollback = getattr(self.storage, "rollback_immediate", None)
        in_txn = callable(begin) and callable(commit)
        if in_txn:
            begin()
        try:
            # Upsert each chunk; capture which (line_start, line_end) we still own
            owned_ranges: set[tuple[int, int]] = set()
            for c, content, emb in zip(chunks, contents, embeddings):
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                chunk_create = ChunkCreate(
                    scope_id=self.scope_id,
                    source_root=self.source_root,
                    source_path=rel,
                    content=content,
                    line_start=c["line_start"],
                    line_end=c["line_end"],
                    language=language,
                    chunk_type="window",
                    metadata={"size_bytes": len(content), "watcher": True},
                )
                try:
                    self.storage.upsert_chunk(
                        chunk_create, content_hash=content_hash, embedding=emb,
                    )
                    owned_ranges.add((c["line_start"], c["line_end"]))
                except Exception as e:
                    logger.warning("watcher: upsert failed for %s:%d-%d: %s",
                                   rel, c["line_start"], c["line_end"], e)
                    self.stats.errors += 1

            # Prune stale chunks (line ranges that existed before this re-chunk)
            try:
                existing = self.storage._conn.execute(
                    "SELECT id, line_start, line_end FROM chunks "
                    "WHERE scope_id = ? AND source_root = ? AND source_path = ?",
                    (self.scope_id, self.source_root, rel),
                ).fetchall()
                stale_ids = [
                    row[0] for row in existing
                    if (row[1], row[2]) not in owned_ranges
                ]
                if stale_ids:
                    placeholders = ",".join("?" * len(stale_ids))
                    self.storage._conn.execute(
                        f"DELETE FROM chunks WHERE id IN ({placeholders})",
                        stale_ids,
                    )
            except Exception as e:
                logger.warning("watcher: stale-prune failed for %s: %s", rel, e)

            if in_txn:
                commit()
        except Exception:
            if in_txn and callable(rollback):
                try:
                    rollback()
                except Exception:
                    pass
            raise

        self.stats.files_reingested += 1
        logger.info("watcher: re-ingested %s (%d chunks)", rel, len(chunks))


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _watchdog_available() -> bool:
    try:
        import watchdog  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# watchdog backend
# ---------------------------------------------------------------------------

class _WatchdogWatcher(_BaseWatcher):
    """Native FSEvents (macOS) / inotify (Linux) backend via the watchdog lib."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher_self = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory:
                    watcher_self._enqueue(Path(event.src_path))
            def on_created(self, event):
                if not event.is_directory:
                    watcher_self._enqueue(Path(event.src_path))
            def on_moved(self, event):
                if not event.is_directory:
                    # treat as delete + create
                    watcher_self._enqueue_deleted(Path(event.src_path))
                    if hasattr(event, "dest_path"):
                        watcher_self._enqueue(Path(event.dest_path))
            def on_deleted(self, event):
                if not event.is_directory:
                    watcher_self._enqueue_deleted(Path(event.src_path))

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self.root), recursive=True)

    def start(self) -> None:
        self._observer.start()
        # Drain loop runs in its own thread
        threading.Thread(
            target=self._drain_loop, name="skein-watcher-drain", daemon=True,
        ).start()

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            self._drain_pending()
            self._stop_event.wait(0.5)

    def stop(self) -> None:
        super().stop()
        try:
            self._observer.stop()
            self._observer.join(timeout=3)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Polling backend (fallback)
# ---------------------------------------------------------------------------

class _PollingWatcher(_BaseWatcher):
    """Pure-Python fallback that scans mtimes every ``poll_interval`` seconds."""

    def __init__(self, *args, poll_interval: float = DEFAULT_POLL_INTERVAL,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.poll_interval = poll_interval
        self._mtimes: dict[Path, float] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._initial_scan()
        self._thread = threading.Thread(
            target=self._loop, name="skein-watcher-poll", daemon=True,
        )
        self._thread.start()

    def _initial_scan(self) -> None:
        for p, mtime in self._scan():
            self._mtimes[p] = mtime

    def _scan(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in self.excludes]
            for fname in filenames:
                p = Path(dirpath) / fname
                if not self._should_track(p):
                    continue
                try:
                    yield p, p.stat().st_mtime
                except OSError:
                    continue

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            current: dict[Path, float] = {}
            for p, mtime in self._scan():
                current[p] = mtime
                prev = self._mtimes.get(p)
                if prev is None or mtime > prev:
                    self._enqueue(p)

            # Detect deletions
            for p in set(self._mtimes) - set(current):
                self._enqueue_deleted(p)

            self._mtimes = current
            self._drain_pending()
            self._stop_event.wait(self.poll_interval)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_watcher(
    root: Path, scope_id: str, source_root: str,
    storage: Storage, provider: EmbeddingProvider | None,
    *, force_polling: bool = False, **kwargs,
) -> _BaseWatcher:
    """Build the best watcher available for this platform / install."""
    if not force_polling and _watchdog_available():
        return _WatchdogWatcher(
            root, scope_id, source_root, storage, provider, **kwargs,
        )
    return _PollingWatcher(
        root, scope_id, source_root, storage, provider, **kwargs,
    )
