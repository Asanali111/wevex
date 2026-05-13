"""Tests for the git commit watcher (iter 15)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import skein.git_watcher as git_watcher
from skein.embeddings import HashEmbeddingProvider
from skein.git_watcher import (
    GitCommit,
    GitCommitWatcher,
    commit_to_fact,
    extract_pr_refs,
    fetch_pr_summary,
    is_noise_commit,
    parse_conventional,
    read_commits_since,
)
from skein.models import IdentityCreate, ScopeCreate
from skein.storage import Storage

# ---------------------------------------------------------------------------
# Conventional Commits parser
# ---------------------------------------------------------------------------


def test_parse_conventional_feat() -> None:
    out = parse_conventional("feat(auth): add OAuth login")
    assert out is not None
    assert out["type"] == "feat"
    assert out["scope"] == "auth"
    assert out["subject"] == "add OAuth login"
    assert out["breaking"] is False


def test_parse_conventional_breaking() -> None:
    out = parse_conventional("refactor!: drop python 3.8 support")
    assert out is not None
    assert out["breaking"] is True


def test_parse_conventional_no_scope() -> None:
    out = parse_conventional("fix: stop hanging on empty stdin")
    assert out["scope"] == ""


def test_parse_conventional_rejects_plain() -> None:
    assert parse_conventional("just a regular commit") is None
    assert parse_conventional("") is None


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------


def _commit(subject: str, body: str = "") -> GitCommit:
    return GitCommit(
        sha="abc1234567",
        author_name="Test",
        author_email="test@example.com",
        timestamp="2026-05-12T10:00:00Z",
        subject=subject,
        body=body,
    )


def test_is_noise_merge() -> None:
    assert is_noise_commit(_commit("Merge branch 'main' into dev"))


def test_is_noise_version_bump() -> None:
    assert is_noise_commit(_commit("bump version to 0.2.0"))


def test_is_noise_wip() -> None:
    assert is_noise_commit(_commit("WIP: trying things"))


def test_is_noise_chore_type() -> None:
    assert is_noise_commit(_commit("chore: update dependencies"))


def test_is_noise_style_type() -> None:
    assert is_noise_commit(_commit("style: reformat with black"))


def test_is_noise_test_type() -> None:
    assert is_noise_commit(_commit("test(retrieval): add hit@5 fixtures for hybrid RRF"))
    assert is_noise_commit(_commit("test: cover empty-storage edge case"))


def test_is_noise_initial_commit() -> None:
    assert is_noise_commit(_commit("Initial commit"))
    assert is_noise_commit(_commit("initial commit"))
    assert is_noise_commit(_commit("INITIAL COMMIT"))
    # Real-world variants from IDE git tools and templates.
    assert is_noise_commit(_commit("Initial commit."))
    assert is_noise_commit(_commit("initial commit of skein"))


def test_docs_adr_not_noise() -> None:
    # docs commits that document architecture decisions ARE real decisions
    # — they're how ADRs land. Don't filter the whole `docs` type.
    assert not is_noise_commit(
        _commit("docs(architecture): document the lease coordination model",
                "ADR-0007. Explains why we picked advisory leases over CRDTs.")
    )


def test_real_commits_not_noise() -> None:
    assert not is_noise_commit(_commit("feat(auth): add SAML login"))
    assert not is_noise_commit(_commit("fix: prevent double-encoding of email subjects"))
    assert not is_noise_commit(_commit("refactor: extract Storage migration helpers"))


# ---------------------------------------------------------------------------
# commit_to_fact
# ---------------------------------------------------------------------------


def test_commit_to_fact_conventional_high_confidence() -> None:
    fact = commit_to_fact(_commit("feat(auth): add OAuth", "Uses Google OAuth2.\nResolves #42."))
    assert fact.type == "decision"
    assert fact.confidence >= 0.9
    assert "feat" in fact.tags
    assert "scope:auth" in fact.tags
    assert "git" in fact.tags
    assert "add OAuth" in fact.content
    assert "Resolves #42" in fact.content
    assert fact.source_file.startswith("git:")


def test_commit_to_fact_non_conventional_lower_confidence() -> None:
    fact = commit_to_fact(_commit("Make the sync command quieter"))
    assert fact.type == "decision"
    assert fact.confidence < 0.9
    assert "commit" in fact.tags


def test_commit_to_fact_truncates_huge_body() -> None:
    big = "x" * 5000
    fact = commit_to_fact(_commit("feat: thing", body=big))
    assert len(fact.content) < 3000
    assert "(truncated)" in fact.content


def test_commit_to_fact_breaking_change_tag() -> None:
    fact = commit_to_fact(_commit("refactor!: drop python 3.8"))
    assert "breaking-change" in fact.tags


# ---------------------------------------------------------------------------
# read_commits_since (requires real git in tmp dir)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_git_repo(tmp_path):
    """Make a real tiny git repo with two commits."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path, check=True)

    (tmp_path / "f1.txt").write_text("a\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "feat: initial commit"], cwd=tmp_path, check=True)

    (tmp_path / "f2.txt").write_text("b\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fix(sync): stop double-write"], cwd=tmp_path, check=True)

    return tmp_path


def test_read_commits_returns_chrono_order(tmp_git_repo: Path) -> None:
    commits = read_commits_since(tmp_git_repo)
    assert len(commits) == 2
    assert commits[0].subject == "feat: initial commit"
    assert commits[1].subject == "fix(sync): stop double-write"


def test_read_commits_since_skips_earlier(tmp_git_repo: Path) -> None:
    all_commits = read_commits_since(tmp_git_repo)
    first_sha = all_commits[0].sha
    later = read_commits_since(tmp_git_repo, since_sha=first_sha)
    assert len(later) == 1
    assert later[0].subject == "fix(sync): stop double-write"


def test_read_commits_returns_empty_for_nongit(tmp_path: Path) -> None:
    assert read_commits_since(tmp_path) == []


# ---------------------------------------------------------------------------
# GitCommitWatcher end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def watcher_setup(tmp_git_repo, tmp_path):
    db = tmp_path / "skein.db"
    storage = Storage(str(db))
    ident = storage.get_or_create_identity(
        IdentityCreate(handle="user:gw-test", type="user", name="t"),
    )
    scope = storage.create_scope(
        ScopeCreate(handle="project:gw", type="project", name="gw", owner_id=ident.id),
    )
    provider = HashEmbeddingProvider()
    w = GitCommitWatcher(
        storage=storage, provider=provider,
        scope_id=scope.id, owner_id=ident.id,
        repo_path=tmp_git_repo,
    )
    yield w, storage, scope, tmp_git_repo
    storage.close()


def test_watcher_poll_promotes_commits_to_fragments(watcher_setup) -> None:
    w, storage, scope, _repo = watcher_setup
    n = w.poll_once()
    assert n == 2  # both commits had conventional prefixes, neither is noise
    frags = storage.list_fragments(scope_id=scope.id, limit=50)
    decisions = [f for f in frags if f.type == "decision" and f.created_by_tool == "git"]
    assert len(decisions) >= 1  # at least the high-conf one auto-promoted
    subjects = " ".join(f.content for f in decisions)
    assert "initial commit" in subjects or "double-write" in subjects


def test_watcher_skips_already_seen_commits(watcher_setup) -> None:
    w, _storage, _scope, repo = watcher_setup
    w.poll_once()
    # Second poll with no new commits = 0
    assert w.poll_once() == 0

    # Add a third commit, only that one should land
    subprocess.run(["git", "commit", "--allow-empty", "-q",
                    "-m", "refactor: rename foo to bar"],
                   cwd=repo, check=True,
                   env={"GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin"})
    assert w.poll_once() == 1


def test_watcher_filters_noise_commits(watcher_setup) -> None:
    w, _storage, _scope, repo = watcher_setup
    subprocess.run(["git", "commit", "--allow-empty", "-q",
                    "-m", "chore: bump version to 9.9.9"],
                   cwd=repo, check=True,
                   env={"GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin"})
    n = w.poll_once()
    # 2 original + 1 noise = 3 commits walked, 2 promoted
    assert n == 2

    # Cursor should still advance past the noise so re-poll = 0
    assert w.poll_once() == 0


# ---------------------------------------------------------------------------
# PR enrichment (Tier 2 #8)
# ---------------------------------------------------------------------------


def test_extract_pr_refs_basic() -> None:
    assert extract_pr_refs("fix: closes #42 and #100") == [42, 100]


def test_extract_pr_refs_dedupes_and_caps() -> None:
    # Repeats are deduped.
    assert extract_pr_refs("#42 fix #42 again, also #42") == [42]
    # > 5 distinct refs are capped at 5.
    refs = extract_pr_refs("#1 #2 #3 #4 #5 #6 #7")
    assert refs == [1, 2, 3, 4, 5]


def test_extract_pr_refs_rejects_inline_hash() -> None:
    assert extract_pr_refs("abc#123") == []
    assert extract_pr_refs("foo.com/x#456") == []


def test_commit_to_fact_pr_enrichment_skipped_without_repo() -> None:
    fact = commit_to_fact(_commit("fix: closes #42", "Some body referencing #42 again."))
    assert "[Linked PRs]" not in fact.content
    assert "pr:#42" not in fact.tags


def test_commit_to_fact_pr_enrichment_with_mock(monkeypatch, tmp_path: Path) -> None:
    canned = {
        "number": 42,
        "title": "Add OAuth",
        "body": "Implements RFC 6749.",
        "url": "https://github.com/x/y/pull/42",
        "state": "MERGED",
    }
    monkeypatch.setattr(git_watcher, "fetch_pr_summary", lambda repo, n: canned)
    fact = commit_to_fact(
        _commit("feat(auth): close #42", "Resolves #42."),
        repo_path=tmp_path,
    )
    assert "[Linked PRs]" in fact.content
    assert "#42: Add OAuth" in fact.content
    assert "https://github.com/x/y/pull/42" in fact.content
    assert "pr:#42" in fact.tags


def test_fetch_pr_summary_returns_none_when_gh_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(git_watcher.shutil, "which", lambda name: None)
    assert fetch_pr_summary(tmp_path, 42) is None
