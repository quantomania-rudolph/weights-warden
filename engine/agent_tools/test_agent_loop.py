import os, sys, json, subprocess, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE); import dispatch as D, agent_loop as AL

def _start(tmp):
    run = str(tmp / "run"); subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "loop", "start", run, "--target-seed", "0"], check=True, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1"}); return run

def test_specs_are_valid_json_schema_in_all_wire_formats_and_hide_finish():
    names = {t["name"] for t in D.TOOLS}; assert "finish" not in names and {"propose", "probe", "note", "state", "weights", "qkv"} <= names
    for t in D.to_openai(): assert t["type"] == "function" and t["function"]["parameters"]["type"] == "object" and all("optional" not in v for v in t["function"]["parameters"]["properties"].values())
    for t in D.to_anthropic(): assert set(t) == {"name", "description", "input_schema"}
    assert D.call("x", "finish", {})["error"].startswith("unknown tool") and "missing" in D.call("x", "probe", {"op": "composite"})["error"]

def test_scripted_vendor_neutral_agent_completes_a_hypothesis_probe_propose_cycle_with_context_reset(tmp_path):
    run = _start(tmp_path); cfg = {"matrix": "dec1.fc1", "axis": "row", "indices": [3, 4], "gamma": 0.5}; script = [
        [("list_ops", {})],
        [("note", {"kind": "hypothesis", "text": "dec1 neurons 3,4 are suspect"}), ("probe", {"op": "scale_slice", "cfg": cfg})],
        [("propose", {"op": "scale_slice", "cfg": cfg}), ("propose", {"op": "scale_slice", "cfg": {**cfg, "gamma": 9.0}}), ("state", {}), ("finish", {})], []]; seen = {"resets": 0}
    def chat(system, msgs, tools):
        assert "finish" not in {t["name"] for t in tools}; seen["resets"] += "CONTEXT RESET" in msgs[0]["content"]; step = script.pop(0)
        return {"text": "ok", "tool_calls": [{"id": f"c{i}", "name": n, "args": a} for i, (n, a) in enumerate(step)]}
    out = AL.run_agent(chat, run, "sys", "task", reset_every=2); assert out["finished"] and seen["resets"] >= 1
    st = D.call(run, "state"); assert st["evaluations_used"] == 1 and st["probes_used"] == 1 and st["notes"] and st["rejected_charged"] == 1, st
    tr = [json.loads(l) for l in open(os.path.join(run, "agent_transcript.jsonl"))]; assert any(r.get("event") == "context_reset" for r in tr) and any(r.get("result_of") == "finish" for r in tr)
