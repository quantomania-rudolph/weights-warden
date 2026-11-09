import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import policy as P, eap_nav as EN

def _m(): torch.manual_seed(0); m = P.NavPolicy(dict(d=32, n_heads=4, n_enc=2, n_dec=2)).eval(); return m, torch.rand(64, 21)

def test_golden_decomposed_forward_equals_model():
    m, o = _m(); out, _ = EN.run(m, o); assert torch.allclose(out, m(o), atol=1e-5)

def test_property_dropping_all_edges_of_a_writer_changes_output_and_unrelated_edge_does_not_exist():
    m, o = _m(); out, c = EN.run(m, o, record=True); es = list(dict.fromkeys(c["edges"])); assert len(es) > 50
    d = EN.run(m, o, frozenset([es[3]]))[0]; assert not torch.allclose(d, out)

def test_property_eap_matches_exact_only_for_tiny_perturbations():
    # first-order EAP is a LINEARISATION: valid for a 0.1% edge perturbation, NOT for whole-edge removal (documented in eap_nav.py; whole edges use exact patching).
    # (Earlier version of this test asserted agreement for whole-edge removal and was stale: it fails, correctly, on this random model.)
    m, o = _m(); y = m(o).detach() + 0.1 * torch.randn(64, 8, 2); v = EN.validate(m, o, y, keep_frac=0.999); assert v["spearman"] > 0.9, v

def test_poison_eap_is_exact_for_a_single_linear_final_edge():
    # dropping an edge into the final reader of the last decoder layer is almost linear: first-order must be close
    m, o = _m(); y = m(o).detach() + 0.2; es, _, _ = EN.eap(m, o, y); ex = EN.exact_edge_scores(m, o, y, [("D1.mlp", "final")])
    assert abs(es[("D1.mlp", "final")] - ex[("D1.mlp", "final")]) < 0.5 * abs(ex[("D1.mlp", "final")]) + 1e-3
