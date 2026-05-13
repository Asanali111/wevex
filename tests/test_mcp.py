"""Tests for the hand-rolled MCP JSON-RPC handler."""
from __future__ import annotations

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mcp(client: TestClient, method: str, params: dict | None = None, req_id: int = 1) -> dict:
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


def test_initialize_includes_recall_first_instructions(client: TestClient) -> None:
    """Q-01: the recall-first guidance is injected via initialize.instructions
    so every MCP client (Claude Code, Cursor, Codex…) sees it in their
    system prompt without needing to GET the prompts/recall-first template.
    """
    result = mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "test-client", "version": "0.1"},
        "capabilities": {},
    })
    instructions = result["result"].get("instructions", "")
    assert instructions, "initialize must return non-empty `instructions`"
    # The recall-first contract must be enforced
    assert "recall" in instructions.lower()
    assert "remember" in instructions.lower()
    # Negative directive against hallucination
    assert "do not invent" in instructions.lower() or "don't invent" in instructions.lower()


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


def test_supersede_marks_old_stale_and_creates_new(client: TestClient) -> None:
    """Q-04: supersede atomically marks the old fragment stale and creates
    a replacement inheriting scope/type/territory/tags."""
    scope = _seed_scope(client, "project:supersede")

    # Remember an outdated fact
    mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {
            "content": "use Redis for the session store",
            "type": "decision",
            "scope": scope,
            "tags": ["infra", "session"],
            "territory": "backend/auth",
        },
    })
    frags = client.get("/v1/fragments", params={"scope": scope}).json()
    assert frags
    old_id = frags[0]["id"]
    old_type = frags[0]["type"]
    old_tags = frags[0]["tags"]
    old_territory = frags[0]["territory"]

    # Supersede with new content
    r = mcp(client, "tools/call", {
        "name": "supersede",
        "arguments": {
            "old_fragment_id": old_id,
            "new_content": "use Memcached for the session store",
            "reason": "team voted to switch in iter 13",
        },
    })
    text = r["result"]["content"][0]["text"]
    assert "Superseded" in text

    # Old fragment must be stale with a pointer to the new one
    old_after = client.get(f"/v1/fragments/{old_id}").json()
    assert old_after["is_stale"] is True
    assert "superseded by" in old_after["stale_reason"]
    assert "team voted to switch" in old_after["stale_reason"]

    # The new fragment must inherit type / tags / territory
    all_frags = client.get(
        "/v1/fragments", params={"scope": scope, "include_stale": "true"}
    ).json()
    new_frags = [f for f in all_frags if f["id"] != old_id]
    assert len(new_frags) == 1
    new_frag = new_frags[0]
    assert new_frag["type"] == old_type
    assert sorted(new_frag["tags"]) == sorted(old_tags)
    assert new_frag["territory"] == old_territory
    assert "Memcached" in new_frag["content"]
    assert new_frag["is_stale"] is False


def test_supersede_unknown_fragment(client: TestClient) -> None:
    r = mcp(client, "tools/call", {
        "name": "supersede",
        "arguments": {
            "old_fragment_id": "does-not-exist",
            "new_content": "irrelevant",
        },
    })
    text = r["result"]["content"][0]["text"]
    assert "not found" in text.lower()


def test_supersede_refuses_already_stale(client: TestClient) -> None:
    """If the old fragment is already stale, supersede should refuse and
    suggest using `remember` directly — chain-superseding clutters history."""
    scope = _seed_scope(client, "project:supersede-stale")
    mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {
            "content": "old fact",
            "type": "fact",
            "scope": scope,
        },
    })
    frag_id = client.get("/v1/fragments", params={"scope": scope}).json()[0]["id"]

    # Mark stale via REST (soft-delete)
    client.delete(f"/v1/fragments/{frag_id}")

    r = mcp(client, "tools/call", {
        "name": "supersede",
        "arguments": {
            "old_fragment_id": frag_id,
            "new_content": "new fact",
        },
    })
    text = r["result"]["content"][0]["text"]
    assert "already stale" in text.lower()


def test_initialize_records_client_name(client: TestClient) -> None:
    """iter 14.0b: ``initialize`` reads ``clientInfo.name`` and stores it
    so subsequent tool calls can attribute writes to the originating tool.
    """
    mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "Claude Code", "version": "1.0"},
        "capabilities": {},
    })
    # Verify it's recorded
    from skein.dependencies import get_storage
    storage = get_storage()
    rows = storage.list_mcp_clients()
    names = {r["client_name"] for r in rows}
    assert "claude-code" in names  # normalized


def test_remember_captures_provenance(client: TestClient) -> None:
    """iter 14.0c: ``remember`` populates created_by_tool from clientInfo."""
    mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "Cursor", "version": "0.40"},
        "capabilities": {},
    })
    scope = _seed_scope(client, "project:prov-test")
    mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {
            "content": "ProvTest decision",
            "type": "decision",
            "scope": scope,
        },
    })
    frags = client.get("/v1/fragments", params={"scope": scope}).json()
    assert frags
    f = frags[0]
    assert f["created_by_tool"] == "cursor"
    assert f["extraction_method"] == "explicit"
    assert f["extraction_confidence"] == 1.0


def test_supersede_records_supersede_chain(client: TestClient) -> None:
    """iter 14.0c: ``supersede`` writes both directions of the chain so
    archaeology can walk forwards and backwards."""
    scope = _seed_scope(client, "project:chain-test")
    mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {"content": "old fact", "type": "fact", "scope": scope},
    })
    old_id = client.get("/v1/fragments", params={"scope": scope}).json()[0]["id"]
    mcp(client, "tools/call", {
        "name": "supersede",
        "arguments": {
            "old_fragment_id": old_id,
            "new_content": "new fact",
        },
    })
    # Find the new one
    all_frags = client.get(
        "/v1/fragments", params={"scope": scope, "include_stale": "true"}
    ).json()
    new_frag = next(f for f in all_frags if f["id"] != old_id)
    old_frag = next(f for f in all_frags if f["id"] == old_id)
    # Chain links should be set in both directions
    assert new_frag["supersedes_fragment_id"] == old_id
    assert old_frag["superseded_by_fragment_id"] == new_frag["id"]


def test_supersede_listed_in_tools(client: TestClient) -> None:
    result = mcp(client, "tools/list")
    tools = {t["name"] for t in result["result"]["tools"]}
    assert "supersede" in tools


def test_mcp_recall_emits_event(client: TestClient, tmp_path, monkeypatch) -> None:
    """R-02: recall/remember/supersede etc. write to the JSONL event log.

    Verifies the integration between the MCP handlers and `events.log_event`.
    """
    import json as _json

    from skein.events import reset_event_logger
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("SKEIN_EVENTS_PATH", str(events_path))
    reset_event_logger()
    try:
        scope = _seed_scope(client, "project:events-test")
        mcp(client, "tools/call", {
            "name": "remember",
            "arguments": {"content": "alpha", "type": "fact", "scope": scope},
        })
        mcp(client, "tools/call", {
            "name": "recall",
            "arguments": {"query": "alpha", "scope": scope},
        })
        assert events_path.exists()
        events = [_json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
        event_types = {e["event"] for e in events}
        assert "remember" in event_types
        assert "recall" in event_types
        # Recall event records the hits count
        recall_evt = next(e for e in events if e["event"] == "recall")
        assert "hits" in recall_evt["details"]
    finally:
        reset_event_logger()


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
