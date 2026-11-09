"""Compact, model-agnostic memory of a run for the hypothesis -> implement -> evaluate -> iterate cycle (so a FRESH context can resume): notes, free probes, charged evaluations, best so far, budget left."""
import os, json


def _read(p): return [json.loads(l) for l in open(p)] if os.path.isfile(p) else []


def digest(run, max_items=12):
    log, res = _read(os.path.join(run, "log.jsonl")), _read(os.path.join(run, "research.jsonl"))
    notes = [{"kind": e["kind"], "text": e["payload"]["text"]} for e in res if e["kind"] in ("hypothesis", "result", "decision")]; probes = [e["payload"] for e in res if e["kind"] == "probe"]; its = [e["payload"] for e in log if e["kind"] == "iterate"]
    best = max(its, key=lambda p: p["metrics"].get("d_success", -9), default=None); cfg = next((e["payload"] for e in log), {}); budget = cfg.get("budget")
    strikes = [e for e in log if e["kind"] in ("strike", "reject", "rejected")]
    return {"budget": budget, "evaluations_used": len(its), "rejected_charged": len(strikes), "probes_used": len(probes),
            "best_charged": ({"op": best["op"], "cfg": best["cfg"], "d_success": round(best["metrics"]["d_success"], 4), "id_drop": round(best["metrics"].get("id_drop", 0), 4), "accepted": best["metrics"].get("accepted")} if best else None),
            "charged_history": [{"i": p["index"], "op": p["op"], "d_success": round(p["metrics"]["d_success"], 4), "id_drop": round(p["metrics"].get("id_drop", 0), 4), "accepted": p["metrics"].get("accepted")} for p in its][-max_items:],
            "recent_probes": [{"op": p["op"], "d_research_l1": (round(p["d_research_l1"], 5) if "d_research_l1" in p else p.get("error"))} for p in probes][-max_items:],
            "notes": notes[-max_items:], "reminder": "negative d_research_l1 = closer to the expert; accept needs d_success >= +0.03 with CI above 0 and small id_drop; random stacks reach ~+0.015 by luck."}
