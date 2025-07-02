from straxen.test_utils import nt_test_run_id

from saltax.test_utils import TEST_DATA_TYPES, get_test_context


def test_make():
    """Test that the contexts and their corresponding data can be created without errors."""

    # Init contexts for both salt and simu modes
    st = {}
    for saltax_mode in ["salt", "simu"]:
        st[saltax_mode] = get_test_context(saltax_mode)

    # Try creating some data_types in both modes
    for dt in TEST_DATA_TYPES:
        for saltax_mode in ["salt", "simu"]:
            st[saltax_mode].make(nt_test_run_id, dt, save=dt)
