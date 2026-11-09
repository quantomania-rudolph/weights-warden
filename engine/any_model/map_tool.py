"""One command to bring ANY model into the harness:
  python any_model/map_tool.py hf:<model-id-or-path> inspect                      module tree + shapes + config (what you read before touching a spec)
  python any_model/map_tool.py hf:<model> propose  --out DIR                      propose a ModelMap spec from shapes/names/config -> DIR/spec.json  (+ warnings)
  python any_model/map_tool.py hf:<model> acclimate --out DIR [--spec DIR/spec.json] [--pairs pairs.json]    PROVE the tools on this model -> DIR/acclimation.json ; exit 1 on any FAIL
  pairs.json: [{"name":..., "clean": "text", "corrupt": "text", "correct": " token", "wrong": " token"}, ...]  (tokenizer-encoded; equal token length required; bad pairs are dropped with reasons)
Python API: spec_map.propose_spec(model) / structure.discover(model, spec) / acclimate.acclimate(model, clean, corrupt, metric_fn, spec=spec).  Torch modules: pass your own pairs as token tensors or (x, x') inputs."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in ("core", "agent_tools", "any_model"): sys.path.insert(0, os.path.join(ROOT, p))
import torch, structure as S, spec_map as SM, acclimate as A, prompt_pairs as PP, patching as P


def load(ref):
    if not ref.startswith("hf:"): raise SystemExit("model must be hf:<id-or-path> (for other torch modules use the Python API)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = ref[3:]; return AutoModelForCausalLM.from_pretrained(name).eval(), (AutoTokenizer.from_pretrained(name) if os.path.exists(os.path.join(name, "tokenizer_config.json")) or not os.path.isdir(name) else None)


def main():
    if len(sys.argv) < 3: sys.exit(__doc__)
    ref, cmd, a = sys.argv[1], sys.argv[2], sys.argv[3:]; opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d; out = opt("--out", "harness_map"); os.makedirs(out, exist_ok=True); model, tok = load(ref)
    if cmd == "inspect": rows, cfg = SM.inspect(model); print("\n".join(rows)); print(json.dumps(cfg)); return
    if cmd == "propose":
        spec, warns = SM.propose_spec(model); json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1); print(json.dumps(spec, indent=1)); [print("WARNING:", w) for w in warns]; return
    if cmd == "acclimate":
        spec = json.load(open(opt("--spec"))) if opt("--spec") else None
        if spec is None:
            try: S.discover(model)
            except S.Unsupported: spec, _ = SM.propose_spec(model); print("no built-in family: using an AUTO-PROPOSED map (it will be verified below)")
        pf = opt("--pairs")
        if pf:
            enc = PP.encode(json.load(open(pf)), tok); keep, rep = PP.check(model, enc); [print("pair", r) for r in rep]
            if not keep: sys.exit("no usable prompt pairs (see reasons above)")
            L = {p["clean"].shape[0] for p in keep}; length = max(L, key=lambda n: sum(p["clean"].shape[0] == n for p in keep)); keep = [p for p in keep if p["clean"].shape[0] == length]; clean, corrupt, mf = PP.batch(keep)
        else:
            g = torch.Generator().manual_seed(0); V = model.config.vocab_size; clean = torch.randint(0, V, (8, 12), generator=g); corrupt = torch.randint(0, V, (8, 12), generator=g); mf = P.logit_diff([1] * 8, [2] * 8); print("no pairs given: mechanics-only acclimation on random tokens")
        rep = A.acclimate(model, clean, corrupt, mf, out_path=os.path.join(out, "acclimation.json"), spec=spec)
        if spec: json.dump(spec, open(os.path.join(out, "spec_used.json"), "w"), indent=1)
        A.show(rep); sys.exit(1 if rep["n_fail"] else 0)
    sys.exit(__doc__)


if __name__ == "__main__": main()
