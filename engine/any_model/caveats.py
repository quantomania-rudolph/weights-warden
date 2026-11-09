"""Short MEASUREMENT caveats attached to LM-mode tool outputs (one `caveat` field per call).  Switch: HARNESS_CAVEATS=0 removes them all (for with/without ablations).
Rules (enforced by tests): one sentence, <= 30 words, NO numbers, and about how far to trust the MEASUREMENT - never a conclusion or a steer about what to edit."""
import os
CAVEATS = {
    "inventory": "Roles are guessed from names and shapes; confirm a role with the weights or a patching test before relying on it.",
    "pairs_check": "A pair passes only if the model prefers the correct answer on clean input; all results depend on how well the pairs isolate the behaviour.",
    "circuit": "Scores come from patching with corrupt activations on your pairs only, and describe necessity; first-order edge scores are approximate, exact ones are not.",
    "grow": "Sufficiency is measured with corrupt-activation patching on your pairs; pruning order is a heuristic and a smaller set is not necessarily the true mechanism.",
    "union": "Shared nodes are only as general as the behaviours and pairs you supplied.",
    "overlap": "Overlap depends on the selection threshold and on the pairs used for each behaviour.",
    "proposals": "Proposals only name nodes from a circuit; whether an edit helps is decided by the charged evaluation, not by this list.",
    "baseline": "Activity is correlational, not causal; heads are summarised by output size, and the map reflects only the prompts sampled.",
    "neighbors": "Co-activation does not imply a causal link; trust a candidate only if the verification beats the random control.",
    "screen": "Scored on research episodes only, so it can mislead about the judge data; a good screen is a reason to spend a charged try, not a result.",
    "sweep": "A curve on research episodes; the best value there is a starting point, and the charged evaluation can disagree.",
    "weights": "Raw values are shown without context; the same number can matter very differently in different matrices.",
    "diff": "Weight change size does not equal functional importance.",
}


def enabled(): return os.environ.get("HARNESS_CAVEATS", "1") != "0"


def attach(cmd, out):
    """add out['caveat'] (dict outputs only) unless disabled."""
    if enabled() and isinstance(out, dict) and cmd in CAVEATS and "error" not in out: out["caveat"] = CAVEATS[cmd]
    return out
