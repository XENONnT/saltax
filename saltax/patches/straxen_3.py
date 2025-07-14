import sys
import inspect
import textwrap
import strax
import straxen

from strax.processing.peak_building import find_peaks
from strax.processing.peak_building import sum_waveform
from straxen.plugins.peaklets.peaklets import peak_saturation_correction
from straxen.plugins.peaklets import Peaklets
from ..utils import replace_source, setattr_module

# objects needs to be present in this namespace for the executed source code
import numpy as np
import numba
from strax import utils
from strax.processing.general import _touching_windows
from strax.dtypes import peak_dtype, DIGITAL_SUM_WAVEFORM_CHANNEL
from strax.processing.peak_building import _build_hit_waveform
from strax.processing.peak_building import store_downsampled_waveform
from straxen.plugins.peaklets.peaklets import _peak_saturation_correction_inner
from ..plugins.records import SCHANNEL_STARTS_AT

mod = find_peaks.__module__
src = inspect.getsource(find_peaks)
olds = [
    """@export
""",
    "cache=True",
    """
        area_per_channel[hit["channel"]] += hit_area_pe
""",
]
news = [
    "",
    "cache=False",
    """
        # Manually shift channels for area_per_channel
        area_per_channel[hit["channel"] % SCHANNEL_STARTS_AT] += hit_area_pe
""",
]
src = replace_source(src, olds, news)
exec(src)
setattr_module(mod, "find_peaks", find_peaks)


mod = sum_waveform.__module__
src = inspect.getsource(sum_waveform)
olds = [
    """@export
""",
    "cache=True",
    """
            ch = h["channel"]
""",
    """
            p["saturated_channel"][ch] |= is_saturated
""",
    """
                if ch < n_top_channels:
""",
    """
            area_per_channel[ch] += area_pe
""",
]
news = [
    "",
    "cache=False",
    """
            # Shift salted channel
            ch = h["channel"]
            ch_shifted = ch % SCHANNEL_STARTS_AT
""",
    """
            p["saturated_channel"][ch_shifted] |= is_saturated
""",
    """
                if ch_shifted < n_top_channels:
""",
    """
            area_per_channel[ch_shifted] += area_pe
""",
]
src = replace_source(src, olds, news)
exec(src)
setattr_module(mod, "sum_waveform", sum_waveform)


mod = peak_saturation_correction.__module__
src = inspect.getsource(peak_saturation_correction)
olds = [
    """Correct the area and per pmt area of peaks from saturation.
""",
    """
            ch = r["channel"]
            if channel_saturated[ch]:
                b_pulse[ch, slice(*b_slice)] += r["data"][slice(*r_slice)]
                b_index[ch, np.argmin(b_index[ch])] = record_i
            else:
                b_sumwf[slice(*b_slice)] += r["data"][slice(*r_slice)] * to_pe[ch]
""",
]
news = [
    """WARNING: This probably doesn't work when we have the salted channel also saturated!!!
    We will be using only the real TPC channels to correct the saturation!!! This is dangerous
    if you are salting things outside WIMP/LowER regions!!!
    Correct the area and per pmt area of peaks from saturation.
""",
    """
            # Shift channels to handle salted channels
            ch = r["channel"]
            ch_shifted = ch % SCHANNEL_STARTS_AT

            if channel_saturated[ch]:
                b_pulse[ch_shifted, slice(*b_slice)] += r["data"][slice(*r_slice)]
                b_index[ch_shifted, np.argmin(b_index[ch_shifted])] = record_i
            else:
                b_sumwf[slice(*b_slice)] += r["data"][slice(*r_slice)] * to_pe[ch]
""",
]
src = replace_source(src, olds, news)
exec(src)
setattr_module(mod, "peak_saturation_correction", peak_saturation_correction)


mod = Peaklets.compute.__module__
src = inspect.getsource(Peaklets.compute)
olds = [
    """
        sorted_hit_channels = hitlets["channel"][hit_max_times_argsort]
""",
]
news = [
    """
        sorted_hit_channels = hitlets["channel"][hit_max_times_argsort] % SCHANNEL_STARTS_AT
""",
]
src = replace_source(src, olds, news)
src = textwrap.dedent(src)
exec(src, sys.modules[mod].__dict__)
sys.modules[mod].__dict__["SCHANNEL_STARTS_AT"] = SCHANNEL_STARTS_AT
Peaklets.compute = sys.modules[mod].__dict__["compute"]


mod = Peaklets.add_hit_features.__module__
src = inspect.getsource(Peaklets.add_hit_features)
olds = [
    """
            peaklet["first_channel"] = sorted_hitlets[0]["channel"]
            peaklet["last_channel"] = sorted_hitlets[-1]["channel"]
""",
]
news = [
    """
            peaklet["first_channel"] = sorted_hitlets[0]["channel"] % SCHANNEL_STARTS_AT
            peaklet["last_channel"] = sorted_hitlets[-1]["channel"] % SCHANNEL_STARTS_AT
""",
]
src = replace_source(src, olds, news)
src = textwrap.dedent(src)
exec(src, sys.modules[mod].__dict__)
sys.modules[mod].__dict__["SCHANNEL_STARTS_AT"] = SCHANNEL_STARTS_AT
Peaklets.add_hit_features = sys.modules[mod].__dict__["add_hit_features"]


del mod, src, olds, news
