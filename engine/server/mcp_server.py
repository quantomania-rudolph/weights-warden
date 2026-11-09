#!/usr/bin/env python3
"""MCP server (stdio, JSON-RPC 2.0) exposing the harness tools to ANY MCP client: Cursor, Claude Code, Claude Desktop, Codex, custom agents.  No third-party dependency.
Tools = the live tool list for the configured mode (harness.env / environment) + `start_run` and `current_run`. Every call goes through session.call -> dispatch.call, the same path as `./h call`.
NOT exposed on purpose: `finish` (the single held-out read) - that is the operator's command (`./h finish`).
Protocol: newline-delimited JSON-RPC on stdin/stdout; logs go to stderr only (stdout is the protocol channel)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent_tools"))
import session as S
SUPPORTED = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_TOOLS = [
    {"name": "start_run", "description": "Create a new editing run (fresh ledger, fixed budget of CHARGED proposals) and make it current. Call once at the start; every other tool acts on the current run.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string", "description": "optional run name [A-Za-z0-9_.-]"}, "budget": {"type": "integer", "description": "charged proposals (default from harness.env)"}}, "additionalProperties": False}},
    {"name": "current_run", "description": "Name of the active run, or null if start_run has not been called.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


def tools():
    return SERVER_TOOLS + [{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]} for t in S.tool_specs()]


def call_tool(name, args):
    if name == "start_run": res = S.start_run(args.get("name"), args.get("budget"))
    elif name == "current_run": res = {"run": S.current_run()}
    else:
        if name not in {t["name"] for t in S.tool_specs()}: return {"error": f"unknown tool {name!r}"}
        res = S.call(name, args)
    return res


def handle(msg):
    """returns a response dict, or None for notifications."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None: return None                                                      # notifications (initialized, cancelled, ...) need no reply
    def ok(r): return {"jsonrpc": "2.0", "id": mid, "result": r}
    def err(code, m): return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": m}}
    if method == "initialize":
        want = params.get("protocolVersion"); ver = want if want in SUPPORTED else SUPPORTED[0]
        return ok({"protocolVersion": ver, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "weight-editing-harness", "version": "1.0"},
                   "instructions": "Weight-editing harness. Call start_run once, then follow AGENTS.md in the repo root: inventory -> read tools -> screen/sweep (free) -> propose (charged). Rejected proposals are charged too."})
    if method == "ping": return ok({})
    if method == "tools/list": return ok({"tools": tools()})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(args, dict): return err(-32602, "arguments must be an object")
        try: res = call_tool(name, args)
        except Exception as e: res = {"error": f"{type(e).__name__}: {e}"}
        return ok({"content": [{"type": "text", "text": json.dumps(res, default=str)}], "isError": isinstance(res, dict) and "error" in res})
    return err(-32601, f"method not found: {method}")


def main():
    os.chdir(S.REPO)                                                                  # relative file arguments (e.g. pairs.json) resolve against the repo root
    print(f"[harness-mcp] repo={S.REPO} mode={os.environ.get('HARNESS_MODE', 'nav')} run={S.current_run()}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: msg = json.loads(line)
        except json.JSONDecodeError: sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n"); sys.stdout.flush(); continue
        if isinstance(msg, list): resp = [r for r in (handle(m) for m in msg) if r is not None] or None
        else: resp = handle(msg)
        if resp is not None: sys.stdout.write(json.dumps(resp) + "\n"); sys.stdout.flush()


if __name__ == "__main__": main()
