import os
import strax
from straxen.test_utils import _get_fake_daq_reader, download_test_data, nt_test_run_id

import saltax


def test_context():
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
        st[saltax_mode] = saltax.contexts.sxenonnt(
            runid=nt_test_run_id,
            saltax_mode=saltax_mode,
            start_end_from_medatata=True,
        )
        # Copied from straxen.test_utils.nt_test_context: https://github.com/XENONnT/straxen/blob/ea3291d32fec284e66dbda66d63cc746bf032494/straxen/test_utils.py#L85  # noqa
        st[saltax_mode].set_config(
            {"diagnose_sorting": True, "diagnose_overlapping": True, "store_per_channel": True}
        )
        st[saltax_mode].register(_get_fake_daq_reader())
        download_test_data(
            "https://raw.githubusercontent.com/XENONnT/"
            "strax_auxiliary_files/"
            "f0d177401e11408b273564f0e29df77528e83d26/"
            "strax_files/"
            "012882-raw_records-z7q2d2ye2t.tar"
        )
        st[saltax_mode].storage = [
            strax.DataDirectory("./strax_test_data", deep_scan=True, provide_run_metadata=True)
        ]
        assert st[saltax_mode].is_stored(nt_test_run_id, "raw_records"), os.listdir(
            st[saltax_mode].storage[-1].path
        )

    # Try creating some data_types in both modes
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
