"""Simulate a sampling-capable MCP client against the server (test mode).

Stdlib only. Drives the full handshake — initialize, notifications/initialized,
tools/list, then a tools/call whose sampling/createMessage request it answers —
against a server started with MCP_TEST_MODE=1. Not collected by pytest; run it
directly against a live server.
"""
import http.client
import json
import threading
import sys

HOST, PORT = "127.0.0.1", 8081
PV = "2025-11-25"
ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  FAIL: {msg}")


def post_json(conn, body, headers):
    conn.request("POST", "/mcp", json.dumps(body), headers)
    return conn.getresponse()


# 1) initialize -----------------------------------------------------------------
print("[1] initialize")
c = http.client.HTTPConnection(HOST, PORT, timeout=15)
r = post_json(c, {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": PV, "capabilities": {"sampling": {}, "elicitation": {}},
               "clientInfo": {"name": "mcp-client-sim", "version": "test"}},
}, {"content-type": "application/json", "accept": "application/json, text/event-stream"})
data = json.loads(r.read())
session_id = r.getheader("mcp-session-id")
print(f"    status={r.status} session={session_id} pv={data['result'].get('protocolVersion')}")
if r.status != 200:
    fail(f"initialize status {r.status}")
if data["result"].get("protocolVersion") != PV:
    fail(f"protocolVersion not echoed (got {data['result'].get('protocolVersion')})")
if not session_id:
    fail("no mcp-session-id response header")
if data["result"]["capabilities"].get("tools") is None:
    fail("no tools capability")
c.close()

H = {"content-type": "application/json",
     "accept": "application/json, text/event-stream",
     "mcp-session-id": session_id or "",
     "mcp-protocol-version": PV}

# 2) notifications/initialized --------------------------------------------------
print("[2] notifications/initialized")
c = http.client.HTTPConnection(HOST, PORT, timeout=15)
r = post_json(c, {"jsonrpc": "2.0", "method": "notifications/initialized"}, H)
r.read()
print(f"    status={r.status} (expect 2xx, empty body)")
if not (200 <= r.status < 300):
    fail(f"notifications/initialized status {r.status}")
c.close()

# 3) tools/list -----------------------------------------------------------------
print("[3] tools/list")
c = http.client.HTTPConnection(HOST, PORT, timeout=15)
r = post_json(c, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, H)
data = json.loads(r.read())
tools = [t["name"] for t in data["result"]["tools"]]
print(f"    status={r.status} tools={tools}")
if "query_rag" not in tools:
    fail("query_rag not advertised")
c.close()

# 4) tools/call with SSE + answer sampling/createMessage ------------------------
print("[4] tools/call (SSE) + sampling round-trip")
c = http.client.HTTPConnection(HOST, PORT, timeout=30)
c.request("POST", "/mcp", json.dumps({
    "jsonrpc": "2.0", "id": 42, "method": "tools/call",
    "params": {"name": "query_rag", "arguments": {"query": "ping test"}},
}), H)
resp = c.getresponse()
ctype = resp.getheader("content-type", "")
print(f"    tools/call status={resp.status} content-type={ctype}")
if "text/event-stream" not in ctype:
    fail(f"tools/call not SSE (content-type={ctype})")

SENTINEL = "PONG-42-OK"


def answer_sampling(req_id):
    """POST the sampling result back on a separate connection, as clients do."""
    cc = http.client.HTTPConnection(HOST, PORT, timeout=15)
    body = {"jsonrpc": "2.0", "id": req_id, "result": {
        "role": "assistant", "content": {"type": "text", "text": SENTINEL},
        "model": "sim-model", "stopReason": "endTurn"}}
    cc.request("POST", "/mcp", json.dumps(body), H)
    rr = cc.getresponse()
    print(f"    -> posted sampling result id={req_id} ack_status={rr.status}")
    rr.read()
    cc.close()


# Read the SSE stream frame by frame.
buf = ""
final = None
sampling_seen = False
fp = resp.fp
while True:
    line = fp.readline()
    if not line:
        break
    line = line.decode("utf-8", "replace")
    if line.startswith(":"):
        continue  # keepalive comment
    if line.strip() == "":
        # end of an event; process buffered data line(s)
        if buf:
            try:
                msg = json.loads(buf)
            except Exception:
                buf = ""
                continue
            buf = ""
            if msg.get("method") == "sampling/createMessage":
                sampling_seen = True
                params = msg.get("params", {})
                print(f"    <- sampling/createMessage id={msg.get('id')} maxTokens={params.get('maxTokens')}")
                if "maxTokens" not in params:
                    fail("sampling/createMessage missing maxTokens")
                # answer on a separate connection (concurrent), as clients do
                threading.Thread(target=answer_sampling, args=(msg["id"],)).start()
            elif "result" in msg and msg.get("id") == 42:
                final = msg
                break
            elif "error" in msg:
                fail(f"tools/call error: {msg['error']}")
                break
        continue
    if line.startswith("data:"):
        buf += line[len("data:"):].strip()

c.close()

if not sampling_seen:
    fail("no sampling/createMessage was emitted")
if final is None:
    fail("no final tools/call result received")
else:
    txt = final["result"]["content"][0]["text"]
    print(f"    <- final result id={final.get('id')} isError={final['result'].get('isError')} text={txt!r}")
    if final.get("id") != 42:
        fail(f"final result id {final.get('id')} != 42")
    if SENTINEL not in txt:
        fail(f"final text did not carry the sampled answer (got {txt!r})")

print()
print("RESULT:", "ALL PROTOCOL CHECKS PASSED" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
