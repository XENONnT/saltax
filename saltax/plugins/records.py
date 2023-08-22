import strax
import straxen

export, __all__ = strax.exporter()
__all__ += ['NO_PULSE_COUNTS']

@export
class PulseProcessing(straxen.PulseProcessing):
    """
    Same PulseProcessing as straxen, but the depends_on have been chanegd to raw_records_simu
    """
    depends_on = 'raw_records_simu'
    