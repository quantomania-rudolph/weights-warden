"""Analyst-facing CLI for the research sandbox.  python nav_sandbox/research_cli.py RUN probe --op OP --cfg JSON   (free; logged; budget 30 probes; teacher-forced L1 change, negative = closer to the expert; NO closed-loop success)
                                                  python nav_sandbox/research_cli.py RUN note --kind hypothesis|result|decision --text '...'   (free, logged research notes)"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import research as R
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("cmd", choices=["probe", "note"]); ap.add_argument("--op"); ap.add_argument("--cfg"); ap.add_argument("--kind"); ap.add_argument("--text")
a = ap.parse_args(); cfg = json.load(open(os.path.join(a.run, "log.jsonl").replace("log.jsonl", "log.jsonl")).readline())["payload"]["target_seed"] if False else None
seed = json.loads(open(os.path.join(a.run, "log.jsonl")).readline())["payload"]["target_seed"]; sb = R.Sandbox(a.run, seed, max_calls=30)
if a.cmd == "probe":
    r = sb.probe(a.op, json.loads(a.cfg)); left = 30 - len([e for e in sb._entries() if e["kind"] == "probe"])
    print(json.dumps({**{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k not in ("op", "cfg")}, "probes_left": left, "base_research_l1": round(sb.base_tf["research"], 4), "base_train_state_l1": round(sb.base_tf["train"], 4)}))
else: sb.note(a.kind, a.text); print("noted")
