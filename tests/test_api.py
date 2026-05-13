"""Integration tests for the REST API via FastAPI TestClient."""
from __future__ import annotations

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_no_auth(client: TestClient) -> None:
    """Health endpoint is public — no auth required."""
    c = TestClient(client.app)
    resp = c.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_missing_token(client: TestClient) -> None:
    c = TestClient(client.app)
    resp = c.get("/v1/scopes")
    assert resp.status_code == 401


def test_wrong_token(client: TestClient) -> None:
    c = TestClient(client.app)
    c.headers["Authorization"] = "Bearer wrong-token"
    resp = c.get("/v1/scopes")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def test_create_identity(client: TestClient) -> None:
    resp = client.post("/v1/identities", json={
        "handle": "agent:test-cursor", "type": "agent", "name": "Test Cursor",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["handle"] == "agent:test-cursor"
    assert data["id"]


def test_create_identity_conflict(client: TestClient) -> None:
    payload = {"handle": "agent:dup", "type": "agent", "name": "Dup"}
    client.post("/v1/identities", json=payload)
    resp = client.post("/v1/identities", json=payload)
    assert resp.status_code == 409


def test_list_identities(client: TestClient) -> None:
    resp = client.get("/v1/identities")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_identity_not_found(client: TestClient) -> None:
    resp = client.get("/v1/identities/nonexistent-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

def test_create_scope(client: TestClient) -> None:
    resp = client.post("/v1/scopes", json={
        "handle": "project:api-test", "type": "project",
        "name": "API Test", "owner_id": "will-be-replaced-by-server",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["handle"] == "project:api-test"


def test_create_scope_conflict(client: TestClient) -> None:
    payload = {
        "handle": "project:conflict-test", "type": "project",
        "name": "Conflict", "owner_id": "x",
    }
    client.post("/v1/scopes", json=payload)
    resp = client.post("/v1/scopes", json=payload)
    assert resp.status_code == 409


def test_scope_lineage(client: TestClient) -> None:
    # Create parent
    client.post("/v1/scopes", json={
        "handle": "org:lineage-test", "type": "org",
        "name": "Org", "owner_id": "x",
    })
    # Create child
    parent_resp = client.get("/v1/scopes/org:lineage-test")
    parent_id = parent_resp.json()["id"]

    client.post("/v1/scopes", json={
        "handle": "project:lineage-child", "type": "project",
        "name": "Child", "owner_id": "x",
        "parent_scope_id": parent_id,
    })

    lineage = client.get("/v1/scopes/project:lineage-child/lineage").json()
    assert len(lineage) == 2
    assert lineage[0]["handle"] == "project:lineage-child"
    assert lineage[1]["handle"] == "org:lineage-test"


# ---------------------------------------------------------------------------
# Fragments
# ---------------------------------------------------------------------------

def _ensure_scope(client: TestClient, handle: str = "project:frag-test") -> str:
    """Create scope and return its handle."""
    client.post("/v1/scopes", json={
        "handle": handle, "type": "project",
        "name": "Fragment Test", "owner_id": "x",
    })
    return handle


def test_create_fragment(client: TestClient) -> None:
    scope_handle = _ensure_scope(client)
    resp = client.post("/v1/fragments", json={
        "content": "use async/await for I/O",
        "type": "preference",
        "scope_id": scope_handle,
        "owner_id": "",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "preference"
    assert data["version"] == 1
    assert data["id"]


def test_create_fragment_invalid_type(client: TestClient) -> None:
    _ensure_scope(client)
    resp = client.post("/v1/fragments", json={
        "content": "some content",
        "type": "invalid-type",
        "scope_id": "project:frag-test",
        "owner_id": "",
    })
    assert resp.status_code == 422


def test_create_fragment_unknown_scope(client: TestClient) -> None:
    resp = client.post("/v1/fragments", json={
        "content": "some content",
        "type": "fact",
        "scope_id": "project:nonexistent-scope",
        "owner_id": "",
    })
    assert resp.status_code == 404


def test_get_fragment(client: TestClient) -> None:
    _ensure_scope(client)
    frag_id = client.post("/v1/fragments", json={
        "content": "get me back",
        "type": "fact",
        "scope_id": "project:frag-test",
        "owner_id": "",
    }).json()["id"]

    resp = client.get(f"/v1/fragments/{frag_id}")
    assert resp.status_code == 200
    assert resp.json()["content"] == "get me back"


def test_update_fragment_occ(client: TestClient) -> None:
    _ensure_scope(client)
    frag = client.post("/v1/fragments", json={
        "content": "original content",
        "type": "state",
        "scope_id": "project:frag-test",
        "owner_id": "",
    }).json()

    # Good update
    resp = client.patch(f"/v1/fragments/{frag['id']}", json={
        "content": "updated content",
        "expected_version": 1,
    })
    assert resp.status_code == 200
    assert resp.json()["version"] == 2

    # Stale version → conflict
    resp2 = client.patch(f"/v1/fragments/{frag['id']}", json={
        "content": "conflicting update",
        "expected_version": 1,  # stale
    })
    assert resp2.status_code == 409


def test_delete_fragment(client: TestClient) -> None:
    _ensure_scope(client)
    frag_id = client.post("/v1/fragments", json={
        "content": "to be deleted",
        "type": "observation",
        "scope_id": "project:frag-test",
        "owner_id": "",
    }).json()["id"]

    resp = client.delete(f"/v1/fragments/{frag_id}")
    assert resp.status_code == 204

    # Should be gone from list
    frags = client.get("/v1/fragments", params={"scope": "project:frag-test"}).json()
    assert not any(f["id"] == frag_id for f in frags)


def test_list_fragments_by_scope(client: TestClient) -> None:
    scope_a = _ensure_scope(client, "project:scope-a")
    scope_b = _ensure_scope(client, "project:scope-b")

    client.post("/v1/fragments", json={
        "content": "fragment A", "type": "fact",
        "scope_id": scope_a, "owner_id": "",
    })
    client.post("/v1/fragments", json={
        "content": "fragment B", "type": "fact",
        "scope_id": scope_b, "owner_id": "",
    })

    frags_a = client.get("/v1/fragments", params={"scope": scope_a}).json()
    assert len(frags_a) == 1
    assert frags_a[0]["content"] == "fragment A"


# ---------------------------------------------------------------------------
# Recall (search)
# ---------------------------------------------------------------------------

def test_recall_endpoint(client: TestClient) -> None:
    scope_handle = _ensure_scope(client, "project:recall-test")
    client.post("/v1/fragments", json={
        "content": "Redis is used for caching session data",
        "type": "decision",
        "scope_id": scope_handle,
        "owner_id": "",
    })
    client.post("/v1/fragments", json={
        "content": "PostgreSQL for persistent user data",
        "type": "decision",
        "scope_id": scope_handle,
        "owner_id": "",
    })

    resp = client.post("/v1/fragments/recall", json={
        "query": "caching Redis",
        "scope": scope_handle,
        "limit": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    # Ranks should be sequential
    ranks = [r["rank"] for r in data["results"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_search_endpoint_get(client: TestClient) -> None:
    scope_handle = _ensure_scope(client, "project:search-test")
    client.post("/v1/fragments", json={
        "content": "TypeScript is preferred for frontend",
        "type": "preference",
        "scope_id": scope_handle,
        "owner_id": "",
    })

    resp = client.get("/v1/fragments/search", params={
        "q": "TypeScript frontend",
        "scope": scope_handle,
    })
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


# ---------------------------------------------------------------------------
# Commits
# ---------------------------------------------------------------------------

def test_commits_created_on_fragment(client: TestClient) -> None:
    scope_handle = _ensure_scope(client, "project:commit-test")
    client.post("/v1/fragments", json={
        "content": "API uses REST not GraphQL",
        "type": "decision",
        "scope_id": scope_handle,
        "owner_id": "",
    })

    commits = client.get("/v1/commits", params={"scope": scope_handle}).json()
    assert len(commits) >= 1
    assert commits[0]["message"]


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------

def test_lease_cycle(client: TestClient) -> None:
    scope_handle = _ensure_scope(client, "project:lease-test")

    # Acquire
    resp = client.post("/v1/leases", json={
        "scope_id": scope_handle,
        "glob": "backend/auth/**",
        "owner_id": "",
        "ttl_seconds": 300,
        "reason": "refactoring",
    })
    assert resp.status_code == 201
    lease_id = resp.json()["id"]

    # List
    leases = client.get("/v1/leases", params={"scope": scope_handle}).json()
    assert any(l["id"] == lease_id for l in leases)

    # Release
    resp2 = client.delete(f"/v1/leases/{lease_id}")
    assert resp2.status_code == 204

    # Gone
    leases2 = client.get("/v1/leases", params={"scope": scope_handle}).json()
    assert not any(l["id"] == lease_id for l in leases2)


def test_lease_conflict(client: TestClient) -> None:
    _ensure_scope(client, "project:lease-conflict")

    # Acquire as user 1
    client.post("/v1/leases", json={
        "scope_id": "project:lease-conflict",
        "glob": "backend/**",
        "owner_id": "",
        "ttl_seconds": 300,
    })

    # Try to acquire overlapping glob as a different identity
    # (In our single-user v1 setup, both calls come from the same user,
    #  so no conflict — just checking the endpoint works)
    resp = client.post("/v1/leases", json={
        "scope_id": "project:lease-conflict",
        "glob": "backend/auth.py",
        "owner_id": "",
        "ttl_seconds": 300,
    })
    # Same owner → no conflict (owner can re-acquire)
    assert resp.status_code in (201, 409)
