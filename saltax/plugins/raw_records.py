import strax
from fuse.plugins.pmt_and_daq.pmt_response_and_daq import PMTResponseAndDAQ

export, __all__ = strax.exporter()


@export
class SPMTResponseAndDAQ(PMTResponseAndDAQ):
    __version__ = "0.0.0"
    provides = "raw_records_simu"
    data_kind = "raw_records_simu"
