from straxen.test_utils import nt_test_context, nt_test_run_id

import saltax


def test_version():
    saltax.__version__


def test_context():
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
