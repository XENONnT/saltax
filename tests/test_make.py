import os
from straxen.test_utils import nt_test_run_id

from saltax.test_utils import TEST_DATA_TYPES, get_test_context


def test_make():
    """Test that the contexts and their corresponding data can be created without errors."""
    if os.environ["NUMBA_DISABLE_JIT"] != "1":
        raise RuntimeError(
            "NUMBA_DISABLE_JIT must be set to 1 to run saltax tests. Because for unknown reasons, "
            "errors of channel number out of range are not raised in numba JIT compiled code. "
            "Please run `export NUMBA_DISABLE_JIT=1` in your terminal before running the tests."
        )

    # Init contexts for both salt and simu modes
    st = {}
    for saltax_mode in ["salt", "simu"]:
        st[saltax_mode] = get_test_context(saltax_mode)

    # Try creating some data_types in both modes
    for dt in TEST_DATA_TYPES:
        for saltax_mode in ["salt", "simu"]:
            st[saltax_mode].make(nt_test_run_id, dt, save=dt)
