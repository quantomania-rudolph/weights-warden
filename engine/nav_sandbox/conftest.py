import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
@pytest.fixture(autouse=True)
def _restore_op_registry():
    """opt-in ops (ridge_head, head_temperature, nudge_weights, scale_slice) must not leak into other tests via the process-global registry."""
    import ops as OPS; keep, hooks = set(OPS.REGISTRY), set(OPS.RANGE_HOOKS); yield
    for k in set(OPS.REGISTRY) - keep: del OPS.REGISTRY[k]
    for k in set(OPS.RANGE_HOOKS) - hooks: del OPS.RANGE_HOOKS[k]
