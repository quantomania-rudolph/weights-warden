"""Launcher so ANY agent (any vendor, any shell-capable harness) reaches the opt-in ops and tools through one entry point.
  python agent_tools/run.py loop     start|ops|propose|status ...    (core/loop.py;  `finish` is the OVERSEER's command, not the analyst's)
  python agent_tools/run.py research RUN probe|note ...              (nav_sandbox/research_cli.py)
  python agent_tools/run.py tools    RUN inventory|neuron_scan|circuit|heads|diff|weights|qkv|attention|activity ...   (agent_tools/analyst_tools.py)
Opt-in ops are enabled unless HARNESS_OPTIN=0.  HARNESS_GENERIC=1 switches to generic mode (any nn.Module; see generic_model.py)."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "nav_planted", "nav_circuits", "nav_sandbox", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
for p in filter(None, os.environ.get("HARNESS_PYTHONPATH", "").split(os.pathsep)): sys.path.insert(0, p)      # the user's own backend module lives here (set in harness.env)
T = {"loop": "core/loop.py", "research": "nav_sandbox/research_cli.py", "tools": "agent_tools/analyst_tools.py"}
if len(sys.argv) < 3 or sys.argv[1] not in T: sys.exit(__doc__)
# ---- backend + mode selection (env vars; default = the Nav2D toy so every old run is unchanged)
#   HARNESS_BACKEND=module:Class   adapter backend (evaluate/forward/probe_obs/load_ckpt), e.g. lm_backend:LMBackend  (root override via the backend's own env var)
#   HARNESS_MODE=llm               HF-style transformer/MLP through a ModelMap (any_model/llm_ops.py): head/neuron ops via the map + matrix-level value ops; Nav-only ops are not registered
#   HARNESS_SPEC=path.json         optional declarative ModelMap (see docs/MODEL_MAP_PROTOCOL.md)
if os.environ.get("HARNESS_BACKEND"):
    import importlib, adapter
    mod, cls = os.environ["HARNESS_BACKEND"].split(":"); adapter.set_backend(getattr(importlib.import_module(mod), cls)())
if os.environ.get("HARNESS_MODE") == "llm":
    import llm_ops, more_ops
    llm_ops.enable(os.environ.get("HARNESS_SPEC") or None); more_ops.enable_value_ops(); more_ops.enable_more_value_ops()
elif os.environ.get("HARNESS_OPTIN", "1") != "0":
    import more_ops, ridge_op; more_ops.enable(); ridge_op.enable()
    if os.environ.get("HARNESS_GENERIC") == "1":
        import generic_model; generic_model.enable_generic()
if os.environ.get("HARNESS_MODE") == "llm" and sys.argv[1] == "tools": T["tools"] = "any_model/llm_cli.py"      # LM-mode read tools
path = os.path.join(ROOT, T[sys.argv[1]]); sys.argv = [path] + sys.argv[2:]; runpy.run_path(path, run_name="__main__")
