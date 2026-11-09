"""Tests for the MCP server and the session layer.  The raw-JSON-RPC test always runs; the official-client test runs only if the `mcp` package is installed (pip install mcp)."""
import os, sys, json, subprocess, shutil, tempfile, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); SERVER = os.path.join(HERE, "mcp_server.py")


def _env(runs):
    e = dict(os.environ); e.update(HARNESS_RUNS=runs, OMP_NUM_THREADS="1", PYTHONWARNINGS="ignore"); e.pop("HARNESS_RUN", None); return e


class Rpc:
    def __init__(self, runs): self.p = subprocess.Popen([sys.executable, SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(runs)); self.i = 0
    def send(self, method, params=None, note=False):
        m = {"jsonrpc": "2.0", "method": method}
        if params is not None: m["params"] = params
        if not note: self.i += 1; m["id"] = self.i
        self.p.stdin.write(json.dumps(m) + "\n"); self.p.stdin.flush()
        return None if note else json.loads(self.p.stdout.readline())
    def tool(self, name, args=None):
        r = self.send("tools/call", {"name": name, "arguments": args or {}})["result"]; return json.loads(r["content"][0]["text"]), r["isError"]
    def close(self): self.p.stdin.close(); self.p.wait(timeout=20)


def test_raw_jsonrpc_session():
    runs = tempfile.mkdtemp(); c = Rpc(runs)
    try:
        r = c.send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})["result"]
        assert r["protocolVersion"] == "2025-06-18" and "tools" in r["capabilities"]
        c.send("notifications/initialized", note=True)
        assert c.send("ping")["result"] == {}
        names = {t["name"] for t in c.send("tools/list")["result"]["tools"]}
        assert {"start_run", "current_run", "inventory", "propose", "status", "screen"} <= names and "finish" not in names
        res, err = c.tool("inventory"); assert err and "no active run" in res["error"]                       # a tool before start_run is a clean error, not a crash
        res, err = c.tool("start_run", {"name": "t1", "budget": 2}); assert not err and res["run"] == "t1" and res["budget"] == 2
        assert c.tool("current_run")[0]["run"] == "t1"
        res, err = c.tool("inventory"); assert not err and res["components"]
        res, err = c.tool("propose", {"op": "scale_head", "cfg": {"layer": 0, "head": 1, "gamma": 2.0}}); assert not err and "accepted" in json.dumps(res) or "verdict" in json.dumps(res)
        res, err = c.tool("status"); assert not err and res["evaluations_left"] == 1                         # the charged proposal is counted
        assert c.tool("finish")[1]                                                                           # not exposed: unknown tool -> isError
        assert "error" in c.send("bogus/method") and "error" in c.send("tools/call", {"name": "propose", "arguments": "x"})
        assert c.tool("start_run", {"name": "t1"})[1]                                                        # refuses to reuse a run name
        assert c.tool("start_run", {"name": "../evil"})[1]                                                   # path traversal in run names is refused
    finally: c.close(); shutil.rmtree(runs, ignore_errors=True)


def test_state_survives_server_restart():
    """the current run is stored on disk, so an editor restarting the server mid-task does not lose the run or reset the budget."""
    runs = tempfile.mkdtemp()
    try:
        c = Rpc(runs); c.send("initialize", {"protocolVersion": "2025-06-18"}); c.tool("start_run", {"name": "keep", "budget": 3}); c.close()
        c = Rpc(runs); c.send("initialize", {"protocolVersion": "2025-06-18"}); assert c.tool("current_run")[0]["run"] == "keep" and c.tool("status")[0]["evaluations_left"] == 3; c.close()
    finally: shutil.rmtree(runs, ignore_errors=True)


def test_official_client():
    pytest.importorskip("mcp"); import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    runs = tempfile.mkdtemp()
    async def go():
        async with stdio_client(StdioServerParameters(command=sys.executable, args=[SERVER], env=_env(runs))) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize(); tl = await s.list_tools(); names = {t.name for t in tl.tools}; assert "start_run" in names and "finish" not in names
                out = await s.call_tool("start_run", {"name": "oc", "budget": 2}); assert not getattr(out, "is_error", getattr(out, "isError", None))
                out = await s.call_tool("inventory", {}); assert not getattr(out, "is_error", getattr(out, "isError", None)) and "components" in out.content[0].text
    try: asyncio.run(go())
    finally: shutil.rmtree(runs, ignore_errors=True)
