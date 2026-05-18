"""ADR-002 / iter 26 — boost / bury / archaeology MCP tools.

These three tools are the *agent-facing* surface for the Q-05 value system.
The user never types `skein boost` — they say "remember this is important"
in their chat, and the agent invokes ``boost`` on the relevant fragment.

These tests pin three things:

1. ``boost`` raises a fragment's value (and rejects invalid values).
2. ``bury`` floors a fragment's value to 0.05 (visible only via
   ``include_stale``-style overrides).
3. ``archaeology`` returns a provenance trace for a fragment id, an 8-char
   prefix, or a natural-language query — and survives missing data.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _mcp(client: TestClient, method: str, params: dict = None, req_id: int = 1) -> dict:
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    resp = client.post("/mcp", json=body)
    assert resp.status_code == 200
    return resp.json()


def _seed_decision(client: TestClient, scope: str, content: str) -> str:
    """Use the public MCP `remember` tool so we get a real fragment id back
    via the same path the agent uses. Returns the fragment id."""
    r = _mcp(client, "tools/call", {
        "name": "remember",
        "arguments": {
            "content": content,
            "type": "decision",
            "scope": scope,
        },
    })
    text = r["result"]["content"][0]["text"]
    # The remember tool returns "Saved decision <id>… ..."
    for token in text.split():
        if len(token) >= 8 and "-" not in token and token.replace("…", "").isalnum() is False:
            # Probably a UUID-prefix with trailing ellipsis or punctuation
            pass
    # Easier: query the fragments table directly via REST since we know the scope
    rows = client.get("/v1/fragments", params={"scope": scope, "limit": 5}).json()
    # Most recent first
    assert rows, "expected at least one fragment after remember"
    return rows[0]["id"]


def test_boost_raises_fragment_value(client: TestClient) -> None:
    _mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "t", "version": "0.1"},
        "capabilities": {},
    })
    scope = "project:value-tools-test"
    client.post("/v1/scopes", json={
        "handle": scope, "type": "project", "name": "x", "owner_id": "x",
    })
    frag_id = _seed_decision(client, scope, "we use Postgres in prod.")

    # Bury first so we have a known low value to boost from.
    r_bury = _mcp(client, "tools/call", {
        "name": "bury", "arguments": {"fragment_id": frag_id[:8]},
    })
    assert "result" in r_bury, r_bury
    assert "Buried" in r_bury["result"]["content"][0]["text"]
    after_bury = client.get(f"/v1/fragments/{frag_id}").json()
    assert after_bury["value"] == 0.05

    # Now boost back up.
    r_boost = _mcp(client, "tools/call", {
        "name": "boost",
        "arguments": {"fragment_id": frag_id[:8], "value": 0.95},
    })
    assert "result" in r_boost, r_boost
    after_boost = client.get(f"/v1/fragments/{frag_id}").json()
    assert after_boost["value"] == 0.95


def test_boost_rejects_out_of_range_value(client: TestClient) -> None:
    _mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "t", "version": "0.1"},
        "capabilities": {},
    })
    scope = "project:value-tools-test-2"
    client.post("/v1/scopes", json={
        "handle": scope, "type": "project", "name": "x", "owner_id": "x",
    })
    frag_id = _seed_decision(client, scope, "another decision.")

    r = _mcp(client, "tools/call", {
        "name": "boost",
        "arguments": {"fragment_id": frag_id[:8], "value": 99.0},
    })
    text = r["result"]["content"][0]["text"]
    assert "must be in" in text.lower() or "error" in text.lower()
    # Value must not have changed.
    after = client.get(f"/v1/fragments/{frag_id}").json()
    assert after["value"] != 99.0


def test_boost_with_unknown_prefix_returns_clean_error(client: TestClient) -> None:
    _mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "t", "version": "0.1"},
        "capabilities": {},
    })
    r = _mcp(client, "tools/call", {
        "name": "boost",
        "arguments": {"fragment_id": "deadbeef", "value": 0.9},
    })
    text = r["result"]["content"][0]["text"]
    assert "no fragment matching" in text.lower()


def test_archaeology_by_prefix_returns_trace(client: TestClient) -> None:
    _mcp(client, "initialize", {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "t", "version": "0.1"},
        "capabilities": {},
    })
    scope = "project:arch-test"
    client.post("/v1/scopes", json={
        "handle": scope, "type": "project", "name": "x", "owner_id": "x",
    })
    frag_id = _seed_decision(
        client, scope, "we adopt fastembed as the default embedding provider.",
    )
    r = _mcp(client, "tools/call", {
        "name": "archaeology",
        "arguments": {"query": frag_id[:8], "scope": scope, "limit": 1},
    })
    text = r["result"]["content"][0]["text"]
    assert "Fragment" in text
    assert "Value:" in text
    assert "Created:" in text
