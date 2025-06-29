from straxen.test_utils import nt_test_context, nt_test_run_id

import saltax

def test_version():
    saltax.__version__


def test_context():
    st_salt = saltax.contexts.sxenonnt(
        runid=nt_test_run_id,
        context=nt_test_context,
        saltax_mode="salt",
    )
    st_simu = saltax.contexts.sxenonnt(
        runid=nt_test_run_id,
        context=nt_test_context,
        saltax_mode="simu",
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
        st_salt.make(nt_test_run_id, dt, save=(dt))
    for dt in dtypes:
        st_simu.make(nt_test_run_id, dt, save=(dt))
