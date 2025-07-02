import os
import pytest


# session scope is needed when running locally to ensure
# that the environment variable is set before any import
@pytest.fixture(autouse=True, scope="session")
def set_env():
    os.environ["NUMBA_DISABLE_JIT"] = "1"


@pytest.fixture(autouse=True, scope="function")
def check_env():
    if os.environ["NUMBA_DISABLE_JIT"] != "1":
        raise RuntimeError(
            "NUMBA_DISABLE_JIT must be set to 1 to run saltax tests. Because for unknown reasons, "
            "errors of channel number out of range are not raised in numba JIT compiled code. "
            "Please run `export NUMBA_DISABLE_JIT=1` in your terminal before running the tests."
        )
