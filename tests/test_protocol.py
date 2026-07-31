"""Protocol-level tests for the MCP endpoint (test mode, no DB/GPU).

Run: MCP_TEST_MODE=1 pytest tests/test_protocol.py
The MCP_TEST_MODE env is set in conftest.py before the app is imported.
"""
import json

import pytest
from fastapi.testclient import TestClient

from src_rag_web import mcp_route
from src_rag_web.mcp_server import app

PV = "2025-11-25"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _init(client):
    r = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": PV},
    }, headers={"accept": "application/json, text/event-stream"})
    return r


def test_health_test_mode(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "mode": "test"}


def test_initialize_echoes_supported_version(client):
    r = _init(client)
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["protocolVersion"] == PV
    assert body["result"]["capabilities"].get("tools") is not None
    assert r.headers.get("mcp-session-id")


def test_initialize_falls_back_on_unsupported_version(client):
    r = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "1999-01-01"},
    }, headers={"accept": "application/json, text/event-stream"})
    assert r.json()["result"]["protocolVersion"] == mcp_route.MCP_PROTOCOL_VERSION_FALLBACK
    assert mcp_route.MCP_PROTOCOL_VERSION_FALLBACK in mcp_route.SUPPORTED_PROTOCOL_VERSIONS


def test_missing_session_id_rejected(client):
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600


def test_notifications_initialized_is_202(client):
    sid = _init(client).headers["mcp-session-id"]
    r = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers={"mcp-session-id": sid})
    assert r.status_code == 202
    assert r.content == b""


def test_tools_list_advertises_query_rag(client):
    sid = _init(client).headers["mcp-session-id"]
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    headers={"mcp-session-id": sid})
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert names == ["query_rag"]


def test_tools_call_requires_event_stream_accept(client):
    sid = _init(client).headers["mcp-session-id"]
    r = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "query_rag", "arguments": {"query": "x"}},
    }, headers={"mcp-session-id": sid, "accept": "application/json"})
    assert r.json()["error"]["code"] == -32600


def test_unknown_tool_name_guarded(client):
    sid = _init(client).headers["mcp-session-id"]
    # Bogus tool is rejected up front with a JSON-RPC error (no stream started).
    r = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "bogus_tool", "arguments": {"query": "x"}},
    }, headers={"mcp-session-id": sid, "accept": "text/event-stream"})
    assert r.json()["error"]["code"] == -32602


def test_unknown_session_id_rejected(client):
    _init(client)  # issue a real session, then use a forged one
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    headers={"mcp-session-id": "11111111-2222-3333-4444-555555555555"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600


def test_body_size_limit(client, monkeypatch):
    monkeypatch.setattr(mcp_route, "_MAX_BODY_BYTES", 100)
    big = "x" * 5000
    r = client.post("/mcp", content=json.dumps({"jsonrpc": "2.0", "id": 1,
                    "method": "initialize", "params": {"pad": big}}),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_concurrency_cap_returns_busy(client, monkeypatch):
    sid = _init(client).headers["mcp-session-id"]
    monkeypatch.setattr(mcp_route, "_MAX_CONCURRENT_TOOL_CALLS", 1)
    monkeypatch.setattr(mcp_route, "_active_tool_calls", 1)  # already at capacity
    r = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "query_rag", "arguments": {"query": "x"}},
    }, headers={"mcp-session-id": sid, "accept": "text/event-stream"})
    assert r.json()["error"]["code"] == -32000


def test_auth_enforced_uniformly(monkeypatch):
    monkeypatch.setattr(mcp_route, "_AUTH_TOKEN", "s3cret")
    with TestClient(app) as c:
        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": PV}}
        assert c.post("/mcp", json=body).status_code == 401
        assert c.post("/mcp", json=body, headers={"authorization": "Bearer nope"}).status_code == 401
        ok = c.post("/mcp", json=body, headers={"authorization": "Bearer s3cret"})
        assert ok.status_code == 200
