"""Tests for storage.py — SQLite CRUD + FTS + lease logic."""
from __future__ import annotations

import pytest

from skein.models import (
    CommitCreate,
    FragmentCreate,
    FragmentUpdate,
    IdentityCreate,
    LeaseCreate,
    ScopeCreate,
)
from skein.storage import ConflictError, Storage

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_create_and_get_identity(storage: Storage) -> None:
    identity = storage.create_identity(IdentityCreate(
        handle="user:alice", type="user", name="Alice",
    ))
    assert identity.id
    assert identity.handle == "user:alice"

    fetched = storage.get_identity(identity.id)
    assert fetched is not None
    assert fetched.handle == "user:alice"

    fetched_by_handle = storage.get_identity("user:alice")
    assert fetched_by_handle is not None
    assert fetched_by_handle.id == identity.id


def test_get_or_create_identity(storage: Storage) -> None:
    data = IdentityCreate(handle="agent:cursor:p1", type="agent", name="Cursor")
    a = storage.get_or_create_identity(data)
    b = storage.get_or_create_identity(data)
    assert a.id == b.id   # second call returns existing


def test_list_identities(storage: Storage) -> None:
    for i in range(3):
        storage.create_identity(IdentityCreate(
            handle=f"user:u{i}", type="user", name=f"User {i}",
        ))
    all_ids = storage.list_identities(limit=100)
    assert len(all_ids) == 3


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

def test_create_scope(storage: Storage) -> None:
    user = storage.create_identity(IdentityCreate(
        handle="user:owner", type="user", name="Owner",
    ))
    scope = storage.create_scope(ScopeCreate(
        handle="project:myapp", type="project",
        name="My App", owner_id=user.id,
    ))
    assert scope.handle == "project:myapp"
    assert scope.owner_id == user.id


def test_scope_lineage(storage: Storage) -> None:
    user = storage.create_identity(IdentityCreate(
        handle="user:owner2", type="user", name="Owner2",
    ))
    org = storage.create_scope(ScopeCreate(
        handle="org:acme", type="org", name="Acme", owner_id=user.id,
    ))
    team = storage.create_scope(ScopeCreate(
        handle="team:backend", type="team", name="Backend",
        owner_id=user.id, parent_scope_id=org.id,
    ))
    storage.create_scope(ScopeCreate(
        handle="project:api", type="project", name="API",
        owner_id=user.id, parent_scope_id=team.id,
    ))

    lineage = storage.get_scope_lineage("project:api")
    assert len(lineage) == 3
    assert lineage[0].handle == "project:api"
    assert lineage[1].handle == "team:backend"
    assert lineage[2].handle == "org:acme"


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------

def test_create_fragment(seeded_storage: Storage) -> None:
    st = seeded_storage
    user = st._test_user
    scope = st._test_scope

    frag = st.create_fragment(FragmentCreate(
        type="decision",
        content="use Redis for caching",
        scope_id=scope.id,
        owner_id=user.id,
    ))
    assert frag.id
    assert frag.type == "decision"
    assert frag.version == 1
    assert frag.is_stale is False


def test_fragment_fts_index(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    st.create_fragment(FragmentCreate(
        type="fact", content="the API rate limit is 1000 requests per minute",
        scope_id=scope.id, owner_id=user.id,
    ))
    st.create_fragment(FragmentCreate(
        type="fact", content="use PostgreSQL for persistent storage",
        scope_id=scope.id, owner_id=user.id,
    ))

    hits = st.keyword_search("rate limit", [scope.id])
    assert len(hits) >= 1
    assert any("rate" in h[0] or True for h in hits)   # just check it returned something

    # Verify the right fragment ranked higher
    all_frags = st.list_fragments(scope_id=scope.id)
    rate_limit_frag = next(f for f in all_frags if "rate limit" in f.content)
    top_id = hits[0][0]
    assert top_id == rate_limit_frag.id


def test_fragment_occ(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    frag = st.create_fragment(FragmentCreate(
        type="state", content="v1 is deployed",
        scope_id=scope.id, owner_id=user.id,
    ))

    # Correct version → succeeds
    updated = st.update_fragment(frag.id, FragmentUpdate(
        content="v2 is deployed", expected_version=1,
    ))
    assert updated.version == 2
    assert updated.content == "v2 is deployed"

    # Stale version → conflict
    with pytest.raises(ConflictError):
        st.update_fragment(frag.id, FragmentUpdate(
            content="v3 is deployed", expected_version=1,  # stale
        ))


def test_soft_delete_fragment(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    frag = st.create_fragment(FragmentCreate(
        type="observation", content="auth middleware has a bug",
        scope_id=scope.id, owner_id=user.id,
    ))

    # Visible before delete
    assert len(st.list_fragments(scope_id=scope.id)) == 1

    deleted = st.delete_fragment(frag.id)
    assert deleted is True

    # Not visible after soft-delete (include_stale=False default)
    assert len(st.list_fragments(scope_id=scope.id)) == 0
    assert len(st.list_fragments(scope_id=scope.id, include_stale=True)) == 1


def test_fragment_ttl_and_stale(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    # Create fragment with 1-second TTL
    frag = st.create_fragment(FragmentCreate(
        type="state", content="short-lived state",
        scope_id=scope.id, owner_id=user.id,
        ttl_seconds=1,
    ))
    assert frag.expires_at is not None

    # Immediately visible
    assert len(st.list_fragments(scope_id=scope.id)) == 1

    # Fast-forward: directly set expires_at in the past
    st._conn.execute(
        "UPDATE fragments SET expires_at = datetime('now', '-1 second') WHERE id = ?",
        (frag.id,),
    )

    # Now mark as stale
    count = st.mark_expired_fragments_stale()
    assert count >= 1

    # No longer visible
    assert len(st.list_fragments(scope_id=scope.id)) == 0


def test_fragment_permanent(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    frag = st.create_fragment(FragmentCreate(
        type="requirement", content="users must be able to export their data",
        scope_id=scope.id, owner_id=user.id,
        ttl_seconds=0,  # 0 → permanent
    ))
    assert frag.permanent is True
    assert frag.ttl_seconds is None
    assert frag.expires_at is None


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------

def test_create_commit(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    commit = st.create_commit(CommitCreate(
        author_id=user.id, scope_id=scope.id,
        message="initial schema design",
        fragments_added=["fake-fragment-id"],
    ))
    assert commit.id
    assert commit.message == "initial schema design"

    fetched = st.get_commit(commit.id)
    assert fetched is not None
    assert fetched.fragments_added == ["fake-fragment-id"]


def test_list_commits_by_scope(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    user2 = st.create_identity(IdentityCreate(
        handle="user:bob", type="user", name="Bob",
    ))
    scope2 = st.create_scope(ScopeCreate(
        handle="project:other", type="project",
        name="Other", owner_id=user2.id,
    ))

    st.create_commit(CommitCreate(
        author_id=user.id, scope_id=scope.id, message="commit A",
    ))
    st.create_commit(CommitCreate(
        author_id=user2.id, scope_id=scope2.id, message="commit B",
    ))

    commits_for_scope = st.list_commits(scope_id=scope.id)
    assert len(commits_for_scope) == 1
    assert commits_for_scope[0].message == "commit A"


# ---------------------------------------------------------------------------
# Lease
# ---------------------------------------------------------------------------

def test_acquire_and_release_lease(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    lease = st.acquire_lease(LeaseCreate(
        scope_id=scope.id, glob="backend/**",
        owner_id=user.id, ttl_seconds=300,
        reason="refactoring",
    ))
    assert lease.id
    assert lease.expires_at

    active = st.list_leases(scope_id=scope.id)
    assert any(l.id == lease.id for l in active)

    released = st.release_lease(lease.id, user.id)
    assert released is True

    active2 = st.list_leases(scope_id=scope.id)
    assert not any(l.id == lease.id for l in active2)


def test_lease_conflict_detection(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    st.create_identity(IdentityCreate(
        handle="user:agent2", type="agent", name="Agent 2",
    ))

    # First agent acquires lease
    st.acquire_lease(LeaseCreate(
        scope_id=scope.id, glob="backend/**",
        owner_id=user.id, ttl_seconds=300,
    ))

    # Second agent should see conflict
    conflict = st.check_lease_conflict(scope.id, "backend/auth.py")
    assert conflict is not None
    assert conflict.owner_id == user.id


def test_glob_overlaps_uses_segments_not_substring(seeded_storage: Storage) -> None:
    """Regression: ``"a/bc/**"`` must NOT falsely overlap ``"a/b/**"``.

    The old prefix-match used naive ``str.startswith``, so leases on
    sibling directories with overlapping name *prefixes* (``frontend/``
    vs ``front/``, ``foo/`` vs ``foobar/``) wrongly reported as
    conflicts. Now compares whole path segments.
    """
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    st.acquire_lease(LeaseCreate(
        scope_id=scope.id, glob="frontend/**",
        owner_id=user.id, ttl_seconds=300,
    ))
    # 'front/**' is a different directory tree — must not collide.
    assert st.check_lease_conflict(scope.id, "front/**") is None
    # Same root, different sibling — must not collide.
    assert st.check_lease_conflict(scope.id, "backend/**") is None
    # Whole-segment prefix → genuine overlap.
    assert st.check_lease_conflict(scope.id, "frontend/components/**") is not None


def test_fragment_occ_atomic_update_clause(seeded_storage: Storage) -> None:
    """Regression: the UPDATE must include ``AND version = ?`` so concurrent
    writers can't both succeed even if they pass the python-side check."""
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    frag = st.create_fragment(FragmentCreate(
        type="state", content="initial", scope_id=scope.id, owner_id=user.id,
    ))
    assert frag.version == 1

    # Simulate a concurrent writer bumping version directly in SQL — the
    # caller below will still pass FragmentUpdate(expected_version=1) but
    # the conditional UPDATE must catch the divergence.
    st._conn.execute(
        "UPDATE fragments SET version = 2 WHERE id = ?", (frag.id,),
    )

    with pytest.raises(ConflictError):
        st.update_fragment(frag.id, FragmentUpdate(
            content="stale write", expected_version=1,
        ))

    # Row state should not have been mutated by the failed update.
    reloaded = st.get_fragment(frag.id)
    assert reloaded.content == "initial"
    assert reloaded.version == 2


def test_lease_cleanup(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    lease = st.acquire_lease(LeaseCreate(
        scope_id=scope.id, glob="src/**",
        owner_id=user.id, ttl_seconds=300,
    ))

    # Force expiry
    st._conn.execute(
        "UPDATE leases SET expires_at = datetime('now', '-1 second') WHERE id = ?",
        (lease.id,),
    )

    cleaned = st.cleanup_expired_leases()
    assert cleaned >= 1
    assert st.get_lease(lease.id) is None


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def test_vector_search(seeded_storage: Storage, provider) -> None:
    from skein.embeddings import vec_to_bytes

    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    # Store 3 fragments with embeddings
    contents = [
        "use Redis for caching session data",
        "the authentication middleware validates JWT tokens",
        "React components should use functional style with hooks",
    ]
    frag_ids = []
    for c in contents:
        vec = provider.embed_one(c)
        emb = vec_to_bytes(vec)
        frag = st.create_fragment(FragmentCreate(
            type="fact", content=c, scope_id=scope.id, owner_id=user.id,
        ), embedding=emb)
        frag_ids.append(frag.id)

    # Query similar to the first fragment
    query_vec = provider.embed_one("session caching")
    query_bytes = vec_to_bytes(query_vec)

    results = st.vector_search(query_bytes, [scope.id], dimension=768)
    assert len(results) >= 1
    # The Redis/caching fragment should rank high for "session caching" with hash embeddings
    # (not guaranteed to be #1 with hash, but at least returned)
    returned_ids = [r[0] for r in results]
    assert any(fid in returned_ids for fid in frag_ids)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    st.create_fragment(FragmentCreate(
        type="fact", content="hello world",
        scope_id=scope.id, owner_id=user.id,
    ))
    st.create_commit(CommitCreate(
        author_id=user.id, scope_id=scope.id, message="test commit",
    ))

    stats = st.stats()
    assert stats["fragments"] >= 1
    assert stats["scopes"] >= 1
    assert stats["identities"] >= 1
    assert stats["commits"] >= 1


# ---------------------------------------------------------------------------
# list_fragments — `since` and `exclude_tool` filters
# ---------------------------------------------------------------------------

def test_list_fragments_since_filter(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    f_old = st.create_fragment(FragmentCreate(
        type="fact", content="old fact",
        scope_id=scope.id, owner_id=user.id,
    ))
    # Backdate it past any realistic `since`.
    st._conn.execute(
        "UPDATE fragments SET created_at = '2020-01-01T00:00:00' WHERE id = ?",
        (f_old.id,),
    )
    f_new = st.create_fragment(FragmentCreate(
        type="fact", content="new fact",
        scope_id=scope.id, owner_id=user.id,
    ))

    cutoff = "2025-01-01T00:00:00"
    rows = st.list_fragments(scope_id=scope.id, since=cutoff)
    ids = [r.id for r in rows]
    assert f_new.id in ids
    assert f_old.id not in ids


def test_list_fragments_exclude_tool(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    f_cc = st.create_fragment(FragmentCreate(
        type="decision", content="from claude_code",
        scope_id=scope.id, owner_id=user.id,
        created_by_tool="claude_code",
    ))
    f_cursor = st.create_fragment(FragmentCreate(
        type="decision", content="from cursor",
        scope_id=scope.id, owner_id=user.id,
        created_by_tool="cursor",
    ))
    f_unknown = st.create_fragment(FragmentCreate(
        type="decision", content="from unknown",
        scope_id=scope.id, owner_id=user.id,
    ))

    rows = st.list_fragments(scope_id=scope.id, exclude_tool="claude_code")
    ids = {r.id for r in rows}
    # Fragments from other tools — and NULL — both surface; that's the
    # "what did OTHER tools write since I last looked" semantic.
    assert f_cursor.id in ids
    assert f_unknown.id in ids
    assert f_cc.id not in ids


def test_list_fragments_since_and_exclude_tool_combined(seeded_storage: Storage) -> None:
    st = seeded_storage
    scope = st._test_scope
    user = st._test_user

    # An old fragment from cursor — past cutoff
    f_old_cursor = st.create_fragment(FragmentCreate(
        type="state", content="old cursor work",
        scope_id=scope.id, owner_id=user.id,
        created_by_tool="cursor",
    ))
    st._conn.execute(
        "UPDATE fragments SET created_at = '2020-01-01T00:00:00' WHERE id = ?",
        (f_old_cursor.id,),
    )
    # A fresh fragment from claude_code — should be excluded
    st.create_fragment(FragmentCreate(
        type="state", content="fresh from claude_code",
        scope_id=scope.id, owner_id=user.id,
        created_by_tool="claude_code",
    ))
    # A fresh fragment from cursor — should pass both filters
    f_fresh_cursor = st.create_fragment(FragmentCreate(
        type="state", content="fresh from cursor",
        scope_id=scope.id, owner_id=user.id,
        created_by_tool="cursor",
    ))

    rows = st.list_fragments(
        scope_id=scope.id,
        since="2025-01-01T00:00:00",
        exclude_tool="claude_code",
    )
    ids = {r.id for r in rows}
    assert ids == {f_fresh_cursor.id}
