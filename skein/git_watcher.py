"""Git commit watcher — turns commits into ``decision`` fragments.

Iter 15: this is the *primary* decision-capture path for Skein.

Why git commits instead of chat transcripts?
- Commits are already curated. The dev wrote the message deliberately.
- They're tied to code. Each commit IS a real diff.
- They're already structured (Conventional Commits: ``feat:``, ``fix:``, ``refactor:``).
- They have a stable hash that doubles as ``created_against_commit``.
- They never produce noise like "per atomic change" the way chat extraction does.

This module replaces the heuristic chat extractor as the main decision feeder.
The transcript watcher stays available as opt-in for users who want it.

Architecture mirrors ``transcript_watcher.py``:

- ``GitCommitWatcher`` — polls one project's commit log, stores each new
  commit as a ``decision`` fragment with full provenance.
- ``MultiProjectGitWatcher`` — walks every Skein-up'd project (those with
  ``.skein/scope`` pin), runs a per-project watcher for each.
- Both use ``transcript_cursors`` (table reused; ``file_path`` becomes the
  ``.git/HEAD`` path of each repo) to track the last-seen commit SHA.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .scanner import ScannedFact

logger = logging.getLogger("skein.git_watcher")


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass
class GitCommit:
    sha: str
    author_name: str
    author_email: str
    timestamp: str          # ISO 8601
    subject: str
    body: str               # full message minus subject line
    files_changed: list[str] = None  # type: ignore


# ---------------------------------------------------------------------------
# Conventional Commits parser
# ---------------------------------------------------------------------------


_CONV_COMMIT_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\((?P<scope>[^)]+)\))?(?P<bang>!)?: (?P<subject>.+)$",
    re.IGNORECASE,
)


def parse_conventional(subject: str) -> Optional[dict[str, str]]:
    """Return type/scope/subject if the message follows Conventional Commits."""
    m = _CONV_COMMIT_RE.match(subject.strip())
    if not m:
        return None
    return {
        "type": m.group("type").lower(),
        "scope": m.group("scope") or "",
        "subject": m.group("subject"),
        "breaking": bool(m.group("bang")),
    }


# Commits we deliberately don't surface as decisions — they're not narrative.
# Note: `docs` is intentionally NOT in this set — ADR-style docs commits
# (e.g. "docs(architecture): document the lease coordination model") ARE
# real decisions and should land in the bus.
_BORING_TYPES = {"chore", "style", "test"}
_BORING_SUBJECT_PATTERNS = [
    re.compile(r"^bump version", re.IGNORECASE),
    re.compile(r"^merge branch", re.IGNORECASE),
    re.compile(r"^merge pull request", re.IGNORECASE),
    re.compile(r"^revert ", re.IGNORECASE),
    re.compile(r"^update (changelog|readme)$", re.IGNORECASE),
    re.compile(r"^wip\b", re.IGNORECASE),
    re.compile(r"^fixup!", re.IGNORECASE),
    re.compile(r"^squash!", re.IGNORECASE),
    re.compile(r"^initial commit\b", re.IGNORECASE),
]


def is_noise_commit(commit: GitCommit) -> bool:
    """Return True for version-bumps, merge commits, WIP, etc."""
    for pat in _BORING_SUBJECT_PATTERNS:
        if pat.match(commit.subject):
            return True
    conv = parse_conventional(commit.subject)
    if conv and conv["type"] in _BORING_TYPES:
        return True
    return False


# ---------------------------------------------------------------------------
# git log parsing
# ---------------------------------------------------------------------------


# Use a unit-separator delimiter that can't appear in commit text — picks
# apart fields without quoting nightmares.
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"
_GIT_FORMAT = _FIELD_SEP.join([
    "%H", "%an", "%ae", "%aI", "%s", "%b",
]) + _RECORD_SEP


def read_commits_since(repo: Path, since_sha: Optional[str] = None,
                        limit: int = 200) -> list[GitCommit]:
    """Return commits in chronological order (oldest first) up to ``limit``.

    Walks every local branch (``--branches``), not just ``HEAD`` — so commits
    on ``feat/*`` and ``experiment/*`` branches reach Skein without first
    being merged to main. Duplicate-SHA-across-branches is naturally collapsed
    by git; duplicate-content-across-commits is collapsed downstream by
    ``passive.promote_scanned_facts`` via content-stem matching.

    If ``since_sha`` is given, exclude commits reachable from it. On a fresh
    repo (no cursor) returns the most recent ``limit`` commits across all
    branches. Returns ``[]`` on any git failure — never raises.
    """
    if not (repo / ".git").exists():
        return []
    args = [
        "git", "-C", str(repo), "log",
        "--branches",
        f"--max-count={limit}",
        "--reverse",
        f"--format={_GIT_FORMAT}",
    ]
    if since_sha:
        args.append(f"^{since_sha}")
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=15.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0:
        # Most likely "fatal: bad revision <since>..HEAD" — the cursor's SHA
        # is no longer in the history (rebase / force-push / shallow clone).
        # Reset to a full read of the most recent N commits.
        if since_sha:
            return read_commits_since(repo, None, limit)
        return []
    commits: list[GitCommit] = []
    for record in out.stdout.split(_RECORD_SEP):
        record = record.strip("\n\r")
        if not record:
            continue
        fields = record.split(_FIELD_SEP)
        if len(fields) < 6:
            continue
        sha, name, email, ts, subject, body = fields[:6]
        commits.append(GitCommit(
            sha=sha, author_name=name, author_email=email,
            timestamp=ts, subject=subject.strip(),
            body=body.strip(),
        ))
    return commits


def files_changed_in(repo: Path, sha: str) -> list[str]:
    """``git diff-tree --no-commit-id --name-only -r <sha>``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "diff-tree",
             "--no-commit-id", "--name-only", "-r", sha],
            capture_output=True, text=True, timeout=5.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0:
        return []
    return [p for p in out.stdout.splitlines() if p.strip()]


# ---------------------------------------------------------------------------
# PR linkage — enrich decision fragments with linked PR data
# ---------------------------------------------------------------------------


_PR_REF_RE = re.compile(r"(?<![A-Za-z0-9])#(\d{1,6})\b")


def extract_pr_refs(text: str) -> list[int]:
    """Return unique PR numbers referenced in ``text`` (capped at 5)."""
    seen: list[int] = []
    for m in _PR_REF_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
        if len(seen) >= 5:
            break
    return seen


def _infer_owner_repo(repo: Path) -> Optional[str]:
    """Parse ``owner/repo`` out of ``remote.origin.url``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    url = out.stdout.strip()
    if not url:
        return None
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@"):
        _, _, tail = url.partition(":")
        slug = tail
    else:
        slug = url.rsplit("/", 2)
        slug = "/".join(slug[-2:]) if len(slug) >= 2 else ""
    parts = [p for p in slug.split("/") if p]
    if len(parts) < 2:
        return None
    return f"{parts[-2]}/{parts[-1]}"


def fetch_pr_summary(repo: Path, pr_number: int) -> Optional[dict]:
    """Return ``{number,title,body,url,state}`` for a PR, or None on any failure."""
    if shutil.which("gh") is None:
        return None
    slug = _infer_owner_repo(repo)
    if not slug:
        return None
    try:
        out = subprocess.run(
            ["gh", "pr", "view", str(pr_number),
             "--json", "title,body,number,url,state",
             "--repo", slug],
            capture_output=True, text=True, timeout=5.0,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Commit → ScannedFact
# ---------------------------------------------------------------------------


def commit_to_fact(commit: GitCommit, repo_path: Optional[Path] = None) -> ScannedFact:
    """Render a commit as the content of a ``decision`` fragment.

    Format:
      <subject>

      <body>            # if non-empty

      [type=feat scope=auth files=3 ...]   # provenance footer
    """
    conv = parse_conventional(commit.subject)
    type_tag = conv["type"] if conv else "commit"

    parts: list[str] = [commit.subject]
    if commit.body:
        parts.append("")
        # Truncate huge bodies so a runaway commit can't drop a 100 KB fragment
        body = commit.body
        if len(body) > 2000:
            body = body[:2000] + "\n…(truncated)"
        parts.append(body)
    content = "\n".join(parts)

    tags = ["git", "commit", type_tag]
    if conv and conv.get("scope"):
        tags.append(f"scope:{conv['scope']}")
    if conv and conv.get("breaking"):
        tags.append("breaking-change")

    if repo_path is not None:
        refs = extract_pr_refs(f"{commit.subject}\n{commit.body}")
        if refs:
            blocks: list[str] = []
            total = 0
            cap = 1500
            truncated = False
            for n in refs:
                pr = fetch_pr_summary(repo_path, n)
                if not pr:
                    continue
                title = pr.get("title", "") or ""
                body = (pr.get("body", "") or "").replace("\r", " ").replace("\n", " ")
                if len(body) > 200:
                    body = body[:200]
                url = pr.get("url", "") or ""
                block = f"- #{n}: {title}\n  {body}\n  {url}"
                if total + len(block) + 1 > cap:
                    truncated = True
                    break
                blocks.append(block)
                total += len(block) + 1
                tags.append(f"pr:#{n}")
            if blocks:
                section = "[Linked PRs]\n" + "\n".join(blocks)
                if truncated:
                    section += "\n…(truncated)"
                content = content + "\n\n" + section

    # Confidence policy:
    #   - Conventional Commits → 0.92 (auto-promotes).
    #   - Non-conv with a substantive body (≥300 chars) → 0.90. The dev wrote
    #     a real explanation, take them at their word even without the prefix.
    #   - Non-conv short → 0.75 (lands in inbox for review).
    # Auto-promote threshold is 0.90 in scanner.classify.
    if conv:
        confidence = 0.92
    elif len(commit.body) >= 300:
        confidence = 0.90
    else:
        confidence = 0.75

    return ScannedFact(
        content=content,
        type="decision",
        confidence=confidence,
        source_file=f"git:{commit.sha[:10]}",
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Single-project watcher
# ---------------------------------------------------------------------------


class GitCommitWatcher:
    """Polls one repo for new commits, promotes each to a decision fragment.

    Cursor is the last-seen commit SHA, stored in the ``transcript_cursors``
    table under the path ``<repo>/.git/HEAD`` with client_name=``git``.
    """

    CURSOR_CLIENT = "git"

    def __init__(
        self,
        *,
        storage,
        provider,
        scope_id: str,
        owner_id: str,
        repo_path: Path,
        source_tool: str = "git",
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.scope_id = scope_id
        self.owner_id = owner_id
        self.repo_path = Path(repo_path).resolve()
        self.source_tool = source_tool

    def _cursor_key(self) -> str:
        return str(self.repo_path / ".git" / "HEAD")

    def _read_cursor(self) -> Optional[str]:
        """Return the last-seen SHA, or None for a fresh repo."""
        key = self._cursor_key()
        row = self.storage._conn.execute(
            "SELECT last_byte_offset FROM transcript_cursors "
            "WHERE file_path = ? AND client_name = ?",
            (key, self.CURSOR_CLIENT),
        ).fetchone()
        if not row:
            return None
        # We piggyback on the existing column. The 40-char SHA gets stored
        # as text in the int column via SQLite's dynamic typing — but cleaner
        # to use a separate table eventually. For now: stored as a single
        # row whose ``last_byte_offset`` IS the sha (cast to int will be 0,
        # so we look it up by file_path only). We store the sha in
        # ``stale_reason``-style — actually let's use a dedicated table.
        # Simpler: store as text by misusing the bytes column.
        # For v1: stash the sha as a side-effect row. Done below.
        return None  # see _last_seen_sha_via_kv

    def _last_seen_sha(self) -> Optional[str]:
        """SHA we processed up to last time. Stored in mcp_clients with a
        synthetic token_prefix scoped to this repo path — pragmatic reuse."""
        key = "git-cursor:" + self._cursor_key()
        row = self.storage._conn.execute(
            "SELECT display_name FROM mcp_clients WHERE token_prefix = ?",
            (key,),
        ).fetchone()
        return row["display_name"] if row else None

    def _set_last_seen_sha(self, sha: str) -> None:
        key = "git-cursor:" + self._cursor_key()
        self.storage.upsert_mcp_client(key, "git-cursor", display_name=sha)

    # ---- core ----

    def poll_once(self) -> int:
        """Process new commits since the last cursor. Returns count promoted."""
        if not (self.repo_path / ".git").exists():
            return 0
        since = self._last_seen_sha()
        commits = read_commits_since(self.repo_path, since_sha=since, limit=200)
        if not commits:
            return 0
        facts: list[ScannedFact] = []
        for commit in commits:
            if is_noise_commit(commit):
                continue
            commit.files_changed = files_changed_in(self.repo_path, commit.sha)
            fact = commit_to_fact(commit, repo_path=self.repo_path)
            # Tag for the promotion pipeline; provenance flows via the
            # source_tool argument so created_against_commit lands cleanly.
            facts.append(fact)

        if facts:
            from .passive import promote_scanned_facts
            for fact, commit in zip(facts, [c for c in commits if not is_noise_commit(c)]):
                # We can't pass arbitrary kwargs through promote_scanned_facts
                # today, but commit_to_fact already encoded the SHA in
                # source_file ("git:<sha10>") and tags. The promotion pipeline
                # will store source_tool="git" for all of them.
                pass
            promote_scanned_facts(
                facts,
                storage=self.storage, provider=self.provider,
                scope_id=self.scope_id, owner_id=self.owner_id,
                source_tool=self.source_tool,
            )
        # Always advance the cursor — even past noise commits — so we don't
        # re-process them on every poll.
        self._set_last_seen_sha(commits[-1].sha)
        return len(facts)


# ---------------------------------------------------------------------------
# Multi-project orchestrator (what the daemon runs)
# ---------------------------------------------------------------------------


def discover_scoped_projects(client_root: Path = None) -> list[Path]:
    """Find every directory that has a Skein scope pin AND a .git folder.

    Mirrors ``transcript_watcher.MultiProjectTranscriptWatcher`` — we discover
    projects via Claude Code's project encoding so the watcher works out of
    the box for anyone using Claude Code, no further config needed.
    """
    from .transcript_watcher import decode_claude_project_dir, default_claude_code_root
    root = client_root or default_claude_code_root()
    found: list[Path] = []
    if not root.is_dir():
        return found
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        path = decode_claude_project_dir(entry.name)
        if not path:
            continue
        if not (path / ".skein" / "scope").is_file():
            continue
        if not (path / ".git").exists():
            continue
        found.append(path)
    return found


class MultiProjectGitWatcher:
    """Polls every Skein-up'd, git-tracked project for new commits."""

    def __init__(
        self,
        *,
        storage_factory,
        provider,
        get_owner_id,
        poll_interval: float = 10.0,
        client_root: Optional[Path] = None,
    ) -> None:
        self.storage_factory = storage_factory
        self.provider = provider
        self.get_owner_id = get_owner_id
        self.poll_interval = poll_interval
        self.client_root = client_root
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="skein-git-watcher", daemon=True,
        )
        self._thread.start()
        logger.info("git commit watcher started; poll=%.1fs", self.poll_interval)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.debug("git watcher poll failed", exc_info=True)
            self._stop.wait(self.poll_interval)

    def poll_once(self) -> dict[str, int]:
        out: dict[str, int] = {}
        storage = self.storage_factory()
        try:
            owner_id = self.get_owner_id(storage)
            for project_path in discover_scoped_projects(self.client_root):
                scope_handle = (project_path / ".skein" / "scope").read_text().strip()
                scope = storage.get_scope(scope_handle)
                if not scope:
                    continue
                w = GitCommitWatcher(
                    storage=storage, provider=self.provider,
                    scope_id=scope.id, owner_id=owner_id,
                    repo_path=project_path,
                )
                n = w.poll_once()
                if n:
                    out[str(project_path)] = n
        finally:
            try:
                storage.close()
            except Exception:
                pass
        return out
