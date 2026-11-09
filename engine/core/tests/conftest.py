import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); LAB4 = os.path.dirname(HERE)
if LAB4 not in sys.path:
    sys.path.insert(0, LAB4)

# ---- analysis-side (ops/rewards/dataroom/peek/loop) tests: stand-in arena with the SAME module names (tests/standin.py)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import pytest, torch
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def backend():
    import adapter, standin
    b = standin.StandinBackend(); adapter.set_backend(b); return b
