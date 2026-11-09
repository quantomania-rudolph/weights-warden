"""Shared session layer used by BOTH front ends: the `./h` command line and the MCP server (engine/server/mcp_server.py).  One place decides: which config file, which run, how a tool is called.
  harness.env  (repo root)   KEY=VALUE lines (HARNESS_BACKEND, HARNESS_MODE, HARNESS_BUDGET, ...).  Real environment variables win over the file.
  runs/<name>/               one directory per editing run (ledger, notes, circuits, screens...).  runs/CURRENT holds the name of the active run; HARNESS_RUN overrides it.
Tools are executed by dispatch.call() exactly as before (subprocess through agent_tools/run.py); this module adds no new behaviour to the engine, only wiring."""
import os, sys, re, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(HERE); REPO = os.path.dirname(ENGINE); sys.path.insert(0, HERE)
ENV_FILE = os.path.join(REPO, "harness.env")


def load_env(path=ENV_FILE):
    """read harness.env into os.environ (without overriding variables that are already set)."""
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    for p in filter(None, os.environ.get("HARNESS_PYTHONPATH", "").split(os.pathsep)): sys.path.insert(0, p if os.path.isabs(p) else os.path.join(REPO, p))
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("PYTHONWARNINGS", "ignore")


load_env()
import dispatch as D                                                                  # after load_env: HARNESS_MODE decides which tool list is active


def runs_dir(): return os.environ.get("HARNESS_RUNS") or os.path.join(REPO, "runs")
def run_path(name): return os.path.join(runs_dir(), name)


def current_run():
    if os.environ.get("HARNESS_RUN"): return os.environ["HARNESS_RUN"]
    f = os.path.join(runs_dir(), "CURRENT"); return open(f).read().strip() if os.path.isfile(f) else None


def start_run(name=None, budget=None):
    """create a new run (a fresh ledger with a budget of CHARGED proposals) and make it current."""
    name = name or time.strftime("run_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name): return {"error": "run name must match [A-Za-z0-9_.-]{1,64}"}
    path = run_path(name)
    if os.path.exists(path): return {"error": f"run {name!r} already exists; choose another name (runs are never overwritten)"}
    os.makedirs(runs_dir(), exist_ok=True); budget = int(budget or os.environ.get("HARNESS_BUDGET", 6))
    p = subprocess.run([D.PY, os.path.join(ENGINE, "agent_tools", "run.py"), "loop", "start", path, "--budget", str(budget)], capture_output=True, text=True, env=os.environ.copy())
    if p.returncode != 0: return {"error": (p.stdout + p.stderr).strip()[-800:]}
    open(os.path.join(runs_dir(), "CURRENT"), "w").write(name); return {"run": name, "budget": budget, "mode": os.environ.get("HARNESS_MODE", "nav"), "next": "call `inventory`, then follow AGENTS.md"}


def call(tool, args=None, run=None):
    """run one tool against the current (or given) run; never raises, returns a JSON-serialisable dict."""
    run = run or current_run()
    if not run: return {"error": "no active run: call start_run first (CLI: ./h start)"}
    if not os.path.isdir(run_path(run)): return {"error": f"run {run!r} does not exist"}
    return D.call(run_path(run), tool, args or {})


def tool_specs(): return D.to_neutral(D.active())


def finish(run=None):
    """the OVERSEER's command: the single held-out read. Not exposed to agents."""
    run = run or current_run()
    if not run: return {"error": "no active run"}
    p = subprocess.run([D.PY, os.path.join(ENGINE, "agent_tools", "run.py"), "loop", "finish", run_path(run)], capture_output=True, text=True, env=os.environ.copy(), timeout=1800)
    out = p.stdout.strip()
    try: return json.loads(out[out.index("{"):])
    except Exception: return {"error": (out + p.stderr)[-800:]}
