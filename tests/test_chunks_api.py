"""REST + MCP integration tests for the chunks (codebase RAG) layer."""
from __future__ import annotations

from skein.embeddings import HashEmbeddingProvider, vec_to_bytes
from skein.models import ChunkCreate, IdentityCreate, ScopeCreate


# ---------------------------------------------------------------------------
# Helpers — seed some chunks via the storage layer (faster than ingesting)
# ---------------------------------------------------------------------------

def _seed(authed_client, app, *, scope_handle="project:codetest"):
    from skein.dependencies import get_storage
    storage = get_storage()
    # Owner identity
    owner = storage.get_or_create_identity(IdentityCreate(
        handle="user:codetest", type="user", name="Code Test",
    ))
    scope = storage.get_or_create_scope(ScopeCreate(
        handle=scope_handle, type="project", name="Code Test", owner_id=owner.id,
    ))
    provider = HashEmbeddingProvider()
    samples = [
        ("src/auth.py", "python", "def login(username, password):\n    return token"),
        ("src/auth.py", "python", "def logout(token):\n    invalidate(token)"),
        ("src/rate_limit.ts", "typescript",
         "export function checkRateLimit(uid: string) { return count < 1000; }"),
        ("docs/README.md", "markdown",
         "# Project\n\nWe use bearer tokens with Authorization header."),
    ]
    for i, (path, lang, content) in enumerate(samples):
        emb = vec_to_bytes(provider.embed_one(content))
        storage.upsert_chunk(
            ChunkCreate(
                scope_id=scope.id,
                source_root="myapp",
                source_path=path,
                content=content,
                line_start=i + 1, line_end=i + 1,
                language=lang,
            ),
            content_hash=f"h-{i}",
            embedding=emb,
        )
    return scope


# ---------------------------------------------------------------------------
# REST: /v1/chunks/search + /v1/chunks/stats
# ---------------------------------------------------------------------------

class TestChunksRESTSearch:
    def test_search_post(self, authed_client, app):
        _seed(authed_client, app)
        resp = authed_client.post(
            "/v1/chunks/search",
            json={"query": "bearer tokens", "scope": "project:codetest", "limit": 5},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] > 0
        # At least one hit should mention bearer/token
        contents = [r["chunk"]["content"].lower() for r in data["results"]]
        assert any("bearer" in c or "token" in c for c in contents)

    def test_search_get(self, authed_client, app):
        _seed(authed_client, app)
        resp = authed_client.get(
            "/v1/chunks/search",
            params={"q": "rate limit", "scope": "project:codetest", "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_search_language_filter(self, authed_client, app):
        _seed(authed_client, app)
        resp = authed_client.post(
            "/v1/chunks/search",
            json={
                "query": "rate limit",
                "scope": "project:codetest",
                "languages": ["typescript"],
                "limit": 5,
            },
        )
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert r["chunk"]["language"] == "typescript"

    def test_stats(self, authed_client, app):
        _seed(authed_client, app)
        resp = authed_client.get(
            "/v1/chunks/stats", params={"scope": "project:codetest"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] >= 4
        assert data["total_files"] >= 3
        assert "python" in data["by_language"]

    def test_list(self, authed_client, app):
        _seed(authed_client, app)
        resp = authed_client.get(
            "/v1/chunks", params={"scope": "project:codetest", "limit": 100},
        )
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 4

    def test_list_with_language_filter(self, authed_client, app):
        _seed(authed_client, app)
        resp = authed_client.get(
            "/v1/chunks",
            params={"scope": "project:codetest", "language": "python"},
        )
        assert resp.status_code == 200
        for c in resp.json():
            assert c["language"] == "python"

    def test_delete_root(self, authed_client, app):
        _seed(authed_client, app)
        # Confirm chunks exist
        list_resp = authed_client.get(
            "/v1/chunks", params={"scope": "project:codetest", "limit": 100},
        )
        assert len(list_resp.json()) >= 4

        del_resp = authed_client.delete(
            "/v1/chunks/myapp", params={"scope": "project:codetest"},
        )
        assert del_resp.status_code == 204

        list_resp2 = authed_client.get(
            "/v1/chunks", params={"scope": "project:codetest", "limit": 100},
        )
        assert list_resp2.json() == []

    def test_unknown_scope_404(self, authed_client, app):
        resp = authed_client.get(
            "/v1/chunks/stats", params={"scope": "project:does-not-exist"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# MCP: search_code tool
# ---------------------------------------------------------------------------

class TestSearchCodeMCP:
    def test_tool_listed(self, authed_client, app):
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        })
        assert resp.status_code == 200
        data = resp.json()
        names = [t["name"] for t in data["result"]["tools"]]
        assert "search_code" in names

    def test_tool_call_with_seeded_chunks(self, authed_client, app):
        _seed(authed_client, app)
        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_code",
                "arguments": {
                    "query": "bearer authentication",
                    "scope": "project:codetest",
                    "limit": 5,
                },
            },
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        text = data["result"]["content"][0]["text"]
        # Iter 31: search_code's response shape changed from "Found N
        # code chunks for X" to "N code chunks for X (top quality=…)" —
        # snippet-by-default rendering plus a quality banner instead of
        # per-result chrome. The chunk count and the query token still
        # appear; we just assert the new shape.
        assert "code chunks for" in text
        assert "bearer" in text.lower() or "token" in text.lower()

    def test_tool_call_no_chunks(self, authed_client, app):
        # Create the scope but no chunks
        from skein.dependencies import get_storage
        storage = get_storage()
        owner = storage.get_or_create_identity(IdentityCreate(
            handle="user:empty", type="user", name="Empty",
        ))
        storage.get_or_create_scope(ScopeCreate(
            handle="project:empty", type="project", name="Empty", owner_id=owner.id,
        ))

        resp = authed_client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_code",
                "arguments": {"query": "anything", "scope": "project:empty"},
            },
        })
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        assert "No code chunks" in text
        assert "skein ingest" in text  # tells the agent how to fix it
