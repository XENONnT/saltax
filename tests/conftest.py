import os
import pytest


@pytest.fixture(autouse=True)
def set_env():
    os.environ["NUMBA_DISABLE_JIT"] = "1"
