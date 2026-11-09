"""REFERENCE agent loop - vendor-neutral.  You supply ONE function:  chat(system_prompt, messages, tools) -> {"text": str, "tool_calls": [{"id": str, "name": str, "args": dict}]}
(wrap any provider's SDK, a local model server, or a script in ~15 lines; to_openai / to_anthropic in dispatch.py give the tool schemas in those wire formats).
Messages are neutral dicts: {"role":"user","content":...} | {"role":"assistant","text":...,"tool_calls":[...]} | {"role":"tool","results":[{"id","name","result"}]}.
AutoResearch cycle: the agent writes hypothesis/result/decision notes (ledger), probes for free, spends charged evaluations, and every `reset_every` turns the context is REPLACED by the task + `state` digest so long runs never lose their memory or overflow.
The loop never exposes `finish` (the held-out read) - the overseer calls `python agent_tools/run.py loop finish RUN` once, afterwards."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import dispatch as D
MAX_RESULT_CHARS = 8000


def run_agent(chat, run, system_prompt, task, max_turns=80, reset_every=None, spec=None):
    spec = spec or D.active()
    names = {t["name"] for t in spec}; msgs = [{"role": "user", "content": task}]; tp = os.path.join(run, "agent_transcript.jsonl"); neutral = D.to_neutral(spec)
    log = lambda rec: open(tp, "a").write(json.dumps(rec, default=str) + "\n")
    for turn in range(max_turns):
        if reset_every and turn and turn % reset_every == 0:
            msgs = [{"role": "user", "content": task + "\n\nCONTEXT RESET - resume from this ledger digest (your own earlier notes included):\n" + json.dumps(D.call(run, "state"))}]; log({"turn": turn, "event": "context_reset"})
        r = chat(system_prompt, msgs, neutral); calls = r.get("tool_calls") or []; msgs.append({"role": "assistant", "text": r.get("text", ""), "tool_calls": calls}); log({"turn": turn, "text": r.get("text", ""), "calls": calls})
        if not calls: return {"finished": True, "turns": turn + 1, "final_text": r.get("text", "")}
        res = []
        for c in calls:
            out = D.call(run, c["name"], c.get("args")) if c["name"] in names else {"error": f"tool {c['name']!r} is not available to the analyst"}
            s = json.dumps(out, default=str); out = out if len(s) <= MAX_RESULT_CHARS else {"truncated": s[:MAX_RESULT_CHARS]}; res.append({"id": c["id"], "name": c["name"], "result": out}); log({"turn": turn, "result_of": c["name"], "chars": len(s)})
        msgs.append({"role": "tool", "results": res})
    return {"finished": False, "turns": max_turns}
