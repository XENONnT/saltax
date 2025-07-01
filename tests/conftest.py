import os
import pytest


# session scope is needed when running locally to ensure
# that the environment variable is set before any import
@pytest.fixture(autouse=True, scope="session")
def set_env():
    os.environ["NUMBA_DISABLE_JIT"] = "1"
