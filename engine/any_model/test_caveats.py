import os, sys, re, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import caveats as CV


def test_caveats_are_short_numberless_and_not_directional():
    for k, v in CV.CAVEATS.items():
        assert len(v.split()) <= 30, k                                       # token budget
        assert not re.search(r"\d", v), k                                    # no numbers (toy results must not leak into prompts)
        assert not re.search(r"\b(should|must|always|never|avoid|prefer)\b", v, re.I), k      # measurement trust only; no steering


def test_every_read_tool_has_a_caveat_and_switch_removes_it(monkeypatch):
    import dispatch as D
    read_tools = {t["name"] for t in D.TOOLS_LLM} - {"list_ops", "note", "propose", "status", "state"}
    assert read_tools <= set(CV.CAVEATS), read_tools - set(CV.CAVEATS)
    monkeypatch.setenv("HARNESS_CAVEATS", "1"); assert CV.attach("grow", {"a": 1})["caveat"] == CV.CAVEATS["grow"]
    assert "caveat" not in CV.attach("grow", {"error": "x"}) and "caveat" not in CV.attach("note", {"noted": True})
    monkeypatch.setenv("HARNESS_CAVEATS", "0"); assert "caveat" not in CV.attach("grow", {"a": 1})
