import os
import pandas as pd
import strax
import straxen
from straxen.test_utils import _get_fake_daq_reader, download_test_data, nt_test_run_id

import saltax
from saltax.utils import straxen_version
from saltax.instructions.generator import instr_file_name, generator_flat

TEST_DATA_TYPES = [
    "microphysics_summary",
    "raw_records_simu",
    "records",
    "peaklets",
    "peak_basics",
    "events",
    "event_basics",
    "event_info",
    "cuts_basic",
]


def get_test_context(saltax_mode):
    """Get a test context for the given saltax mode."""
    st = saltax.contexts.sxenonnt(
        runid=nt_test_run_id,
        saltax_mode=saltax_mode,
        # lowest possible version to modify as less as possible
        corrections_version="global_v10",
    )
    st.apply_xedocs_configs(version="global_ONLINE")
    # Patch tf_model_mlp to be compatible with keras version
    if straxen_version() == 3:
        st.set_config(
            {"tf_model_mlp": straxen.PeakletPositionsMLP.takes_config["tf_model_mlp"].default}
        )
    # Copied from straxen.test_utils.nt_test_context: https://github.com/XENONnT/straxen/blob/ea3291d32fec284e66dbda66d63cc746bf032494/straxen/test_utils.py#L85  # noqa
    st.set_config(
        {"diagnose_sorting": True, "diagnose_overlapping": True, "store_per_channel": True}
    )
    st.register(_get_fake_daq_reader())
    download_test_data(
        "https://raw.githubusercontent.com/XENONnT/"
        "strax_auxiliary_files/"
        "f0d177401e11408b273564f0e29df77528e83d26/"
        "strax_files/"
        "012882-raw_records-z7q2d2ye2t.tar"
    )
    st.storage = [
        strax.DataDirectory("./strax_test_data", deep_scan=True, provide_run_metadata=True)
    ]
    assert st.is_stored(nt_test_run_id, "raw_records"), os.listdir(st.storage[-1].path)
    st.set_config({"input_file": generate_instr(st)})
    return st


def generate_instr(context):
    """Generate an instruction file for a given runid and test context."""
    input_file = instr_file_name(
        runid=nt_test_run_id,
    )
    instr = generator_flat(
        runid=nt_test_run_id,
        context=context,
        start_end_from_medatata=True,
    )
    pd.DataFrame(instr).to_csv(input_file, index=False)
    return input_file
