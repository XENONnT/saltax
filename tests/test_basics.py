import os
from straxen.test_utils import nt_test_context, nt_test_run_id

import saltax


def test_context():
    """Test that the contexts and their corresponding data can be created without errors."""
    if os.environ["NUMBA_DISABLE_JIT"] != "1":
        raise RuntimeError(
            "NUMBA_DISABLE_JIT must be set to 1 to run saltax tests. Because for unknown reasons, "
            "errors of channel number out of range are not raised in numba JIT compiled code. "
            "Please run `export NUMBA_DISABLE_JIT=1` in your terminal before running the tests."
        )
    st = {}
    for saltax_mode in ["salt", "simu"]:
        st[saltax_mode] = saltax.contexts.sxenonnt(
            runid=nt_test_run_id,
            context=nt_test_context,
            corrections_version=None,
            run_without_proper_corrections=True,
            saltax_mode=saltax_mode,
            output_folder=None,
            start_end_from_medatata=True,
        )
    dtypes = [
        "microphysics_summary",
        "raw_records_simu",
        "records",
        "peaklets",
        "peak_basics",
        "events",
        "event_basics",
        "event_info",
    ]
    for dt in dtypes:
        for saltax_mode in ["salt", "simu"]:
            st[saltax_mode].make(nt_test_run_id, dt, save=dt)
