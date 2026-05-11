"""Tests for the hand-rolled MCP JSON-RPC handler."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mcp(client: TestClient, method: str, params: dict = None, req_id: int = 1) -> dict:
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    resp = client.post("/mcp", json=body)
    assert resp.status_code == 200
    return resp.json()


def _seed_scope(client: TestClient, handle: str = "project:mcp-test") -> str:
    client.post("/v1/scopes", json={
        "handle": handle, "type": "project",
        "name": "MCP Test", "owner_id": "x",
    })
    return handle


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_initialize(client: TestClient) -> None:
    result = mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "test-client", "version": "0.1"},
        "capabilities": {},
    })
    assert "result" in result
    assert result["result"]["protocolVersion"] == "2024-11-05"
    assert result["result"]["serverInfo"]["name"] == "skein"


def test_unknown_method(client: TestClient) -> None:
    result = mcp(client, "foo/bar")
    assert "error" in result
    assert result["error"]["code"] == -32601


def test_notification_no_response(client: TestClient) -> None:
    """Notifications (no id field) should return 202 with null body."""
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------

def test_tools_list(client: TestClient) -> None:
    result = mcp(client, "tools/list")
    tools = result["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "recall" in tool_names
    assert "remember" in tool_names
    assert "note_decision" in tool_names
    assert "claim_lease" in tool_names
    assert "release_lease" in tool_names
    assert "query_leases" in tool_names
    assert "recall_one" in tool_names


def test_tool_schemas_have_required_inputSchema(client: TestClient) -> None:
    result = mcp(client, "tools/list")
    for tool in result["result"]["tools"]:
        assert "inputSchema" in tool, f"{tool['name']} missing inputSchema"
        assert "properties" in tool["inputSchema"], f"{tool['name']} missing properties"


# ---------------------------------------------------------------------------
# remember / recall
# ---------------------------------------------------------------------------

def test_remember_and_recall(client: TestClient) -> None:
    scope = _seed_scope(client)

    # remember
    r = mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {
            "content": "use async/await for all network I/O",
            "type": "preference",
            "scope": scope,
        },
    })
    assert "result" in r
    assert "content" in r["result"]
    assert "Stored" in r["result"]["content"][0]["text"]

    # recall
    r2 = mcp(client, "tools/call", {
        "name": "recall",
        "arguments": {
            "query": "async programming patterns",
            "scope": scope,
        },
    })
    assert "result" in r2
    text = r2["result"]["content"][0]["text"]
    assert "async" in text.lower() or "Found" in text


def test_remember_auto_creates_unknown_scope(client: TestClient) -> None:
    """Iteration 11: MCP `remember` now auto-creates the scope on first use.

    The old behaviour ("Scope X not found, create it first") was a needless
    speed bump for the AI — it had to call back to the human to run
    `skein scope create` before remembering anything. The auto-create path
    is bounded by ``_ensure_scope`` which only creates well-shaped handles."""
    r = mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {
            "content": "some content",
            "type": "fact",
            "scope": "project:auto-created",
        },
    })
    text = r["result"]["content"][0]["text"]
    assert "Stored fragment" in text


def test_recall_auto_resolves_missing_scope(client: TestClient) -> None:
    """Iteration 11: MCP `recall` may be called without a `scope` arg —
    Skein resolves it from the daemon's cwd."""
    r = mcp(client, "tools/call", {
        "name": "recall",
        "arguments": {"query": "anything"},
    })
    # Either no results or a 'Found' answer — never an error message.
    text = r["result"]["content"][0]["text"]
    assert text and "Error" not in text and "error" not in text


def test_recall_empty_scope(client: TestClient) -> None:
    _seed_scope(client, "project:empty-mcp")
    r = mcp(client, "tools/call", {
        "name": "recall",
        "arguments": {"query": "anything", "scope": "project:empty-mcp"},
    })
    text = r["result"]["content"][0]["text"]
    assert "No relevant" in text or "Found 0" in text or "Found" in text


def test_recall_one(client: TestClient) -> None:
    scope = _seed_scope(client, "project:recall-one")
    # Remember a fragment
    mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {
            "content": "PostgreSQL is used for the user table",
            "type": "fact",
            "scope": scope,
        },
    })
    # Get the fragment ID from REST
    frags = client.get("/v1/fragments", params={"scope": scope}).json()
    assert frags
    frag_id = frags[0]["id"]

    r = mcp(client, "tools/call", {
        "name": "recall_one",
        "arguments": {"fragment_id": frag_id},
    })
    text = r["result"]["content"][0]["text"]
    assert "PostgreSQL" in text


def test_note_decision(client: TestClient) -> None:
    scope = _seed_scope(client, "project:note-decision")
    r = mcp(client, "tools/call", {
        "name": "note_decision",
        "arguments": {
            "content": "use Kafka for event streaming",
            "scope": scope,
            "alternatives": "RabbitMQ, Redis Pub/Sub",
            "rationale": "Kafka has better durability guarantees",
        },
    })
    text = r["result"]["content"][0]["text"]
    assert "recorded" in text.lower() or "Decision" in text

    # Verify the full decision content was stored
    frags = client.get("/v1/fragments", params={"scope": scope}).json()
    assert frags
    stored = frags[0]["content"]
    assert "Kafka" in stored
    assert "RabbitMQ" in stored
    assert "durability" in stored


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------

def test_claim_and_release_lease(client: TestClient) -> None:
    scope = _seed_scope(client, "project:lease-mcp")

    r = mcp(client, "tools/call", {
        "name": "claim_lease",
        "arguments": {
            "glob": "src/auth/**",
            "scope": scope,
            "ttl_seconds": 300,
            "reason": "refactoring auth",
        },
    })
    text = r["result"]["content"][0]["text"]
    assert "acquired" in text.lower()

    # Extract lease ID from text
    import re
    m = re.search(r"([0-9a-f]{8})…", text)
    assert m, "Lease ID not found in response"
    lease_prefix = m.group(1)

    # Query leases
    r2 = mcp(client, "tools/call", {
        "name": "query_leases",
        "arguments": {"scope": scope},
    })
    leases_text = r2["result"]["content"][0]["text"]
    assert lease_prefix in leases_text

    # Release
    leases_list = client.get("/v1/leases", params={"scope": scope}).json()
    assert leases_list
    full_lease_id = leases_list[0]["id"]

    r3 = mcp(client, "tools/call", {
        "name": "release_lease",
        "arguments": {"lease_id": full_lease_id},
    })
    text3 = r3["result"]["content"][0]["text"]
    assert "released" in text3.lower()


# ---------------------------------------------------------------------------
# resources/list + resources/read
# ---------------------------------------------------------------------------

def test_resources_list_is_empty(client: TestClient) -> None:
    """Per the MCP spec, `resources/list` enumerates concrete resources only.
    Skein's context URIs are templates (parameterised by scope), so they live
    under `resources/templates/list` and `resources/list` returns []."""
    result = mcp(client, "resources/list")
    assert result["result"]["resources"] == []


def test_resources_templates_list(client: TestClient) -> None:
    result = mcp(client, "resources/templates/list")
    templates = result["result"]["resourceTemplates"]
    uris = [t["uriTemplate"] for t in templates]
    assert any("state" in u for u in uris)
    assert any("decisions" in u for u in uris)
    assert any("agents-md" in u for u in uris)
    assert any("recent-commits" in u for u in uris)
    # Templates expose `{scope}` placeholders (RFC 6570).
    assert all("{scope}" in u for u in uris)


def test_read_agents_md_resource(client: TestClient) -> None:
    scope = _seed_scope(client, "project:resource-test")
    result = mcp(client, "resources/read", {
        "uri": f"context://{scope}/agents-md",
    })
    contents = result["result"]["contents"]
    assert contents
    assert "AGENTS.md" in contents[0]["text"] or "Skein" in contents[0]["text"]


def test_read_decisions_resource(client: TestClient) -> None:
    scope = _seed_scope(client, "project:decisions-resource")
    client.post("/v1/fragments", json={
        "content": "use REST not GraphQL",
        "type": "decision",
        "scope_id": scope,
        "owner_id": "",
    })
    result = mcp(client, "resources/read", {
        "uri": f"context://{scope}/decisions",
    })
    text = result["result"]["contents"][0]["text"]
    assert "REST" in text


# ---------------------------------------------------------------------------
# prompts/list + prompts/get
# ---------------------------------------------------------------------------

def test_prompts_list(client: TestClient) -> None:
    result = mcp(client, "prompts/list")
    prompts = result["result"]["prompts"]
    names = {p["name"] for p in prompts}
    assert "session_start" in names
    assert "recall-first" in names


def test_get_recall_first_prompt(client: TestClient) -> None:
    result = mcp(client, "prompts/get", {
        "name": "recall-first",
        "arguments": {"scope": "project:test"},
    })
    messages = result["result"]["messages"]
    text = messages[0]["content"]["text"]
    # Mandatory recall semantics
    assert "recall" in text.lower()
    assert "remember" in text.lower()
    # Scope is woven in when provided
    assert "project:test" in text


def test_get_recall_first_prompt_no_scope(client: TestClient) -> None:
    result = mcp(client, "prompts/get", {
        "name": "recall-first",
        "arguments": {},
    })
    text = result["result"]["messages"][0]["content"]["text"]
    assert "recall" in text.lower()


def test_get_session_start_prompt(client: TestClient) -> None:
    scope = _seed_scope(client, "project:prompt-test")
    result = mcp(client, "prompts/get", {
        "name": "session_start",
        "arguments": {"scope": scope, "task": "implement user auth"},
    })
    assert "messages" in result["result"]
    messages = result["result"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    text = messages[0]["content"]["text"]
    assert "AGENTS.md" in text or "Skein" in text


def test_get_unknown_prompt(client: TestClient) -> None:
    result = mcp(client, "prompts/get", {"name": "no-such-prompt"})
    assert "error" in result


# ---------------------------------------------------------------------------
# Batch requests
# ---------------------------------------------------------------------------

def test_batch_requests(client: TestClient) -> None:
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    resp = client.post("/mcp", json=batch)
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list)
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {1, 2}


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------

def test_parse_error(client: TestClient) -> None:
    resp = client.post("/mcp", content=b"not valid json",
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"]["code"] == -32700


# ---------------------------------------------------------------------------
# Auth — regression: the MCP endpoint must validate the bearer token.
# Before the fix, any local process could call recall/remember without auth.
# ---------------------------------------------------------------------------

def test_mcp_endpoint_rejects_missing_auth(app) -> None:
    """No Authorization header → 401, even before JSON parsing."""
    with TestClient(app, raise_server_exceptions=True) as c:
        # Note: no Authorization header set on this client.
        resp = c.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
        })
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate", "").startswith("Bearer")


def test_mcp_endpoint_rejects_wrong_token(app) -> None:
    with TestClient(app, raise_server_exceptions=True) as c:
        c.headers["Authorization"] = "Bearer this-is-not-the-right-token"
        resp = c.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        })
    assert resp.status_code == 401


def test_mcp_endpoint_rejects_unauthenticated_tool_call(app) -> None:
    """Without auth, you can't call dangerous tools — claim_lease, remember."""
    with TestClient(app, raise_server_exceptions=True) as c:
        resp = c.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {
                "name": "remember",
                "arguments": {"content": "x", "type": "fact"},
            },
        })
    assert resp.status_code == 401
