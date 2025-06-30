import inspect
import textwrap
import numpy as np

import strax
import straxen
from straxen.plugins.peaklets.peaklets import Peaklets
from straxen.plugins.peaklets.peaklets import peak_saturation_correction

import numba  # noqa: F401
from strax import utils  # noqa: F401
from strax.processing.general import _touching_windows  # noqa: F401
from strax.dtypes import peak_dtype, DIGITAL_SUM_WAVEFORM_CHANNEL  # noqa: F401
from strax.processing.peak_building import _build_hit_waveform  # noqa: F401
from strax.processing.peak_building import store_downsampled_waveform  # noqa: F401
from straxen.plugins.peaklets.peaklets import hit_max_sample, get_tight_coin  # noqa: F401
from straxen.plugins.peaklets.peaklets import drop_data_top_field  # noqa: F401
from straxen.plugins.peaklets.peaklets import _peak_saturation_correction_inner  # noqa: F401

from ..utils import replace_source
from .records import SCHANNEL_STARTS_AT


export, __all__ = strax.exporter()


@export
class SPeaklets(straxen.Peaklets):
    __version__ = "0.0.4"

    schannel_starts_at = straxen.URLConfig(
        default=SCHANNEL_STARTS_AT,
        infer_type=False,
        help="Salting channel starts at this channel",
    )

    saltax_mode = straxen.URLConfig(
        default="salt", track=True, infer_type=False, help="'data', 'simu', or 'salt'"
    )

    gain_model_mc = straxen.URLConfig(
        infer_type=False, help="PMT gain model. Specify as URL or explicit value"
    )

    def setup(self):
        super().setup()

        # Get the gain model
        # It should have length 3494 for XENONnT
        self.to_pe_data = self.gain_model
        self.to_pe_simu = self.gain_model_mc
        self.to_pe = np.zeros(self.schannel_starts_at + self.n_tpc_pmts, dtype=np.float64)
        self.to_pe[: self.n_tpc_pmts] = self.to_pe_data
        self.to_pe[self.schannel_starts_at :] = self.to_pe_simu

        # Get the hitfinder thresholds
        self.hit_thresholds = np.zeros(self.schannel_starts_at + self.n_tpc_pmts, dtype=np.int64)
        self.hit_thresholds[: self.n_tpc_pmts] = self.hit_min_amplitude
        self.hit_thresholds[self.schannel_starts_at :] = self.hit_min_amplitude

        self.channel_range = (
            min(min(self.channel_map["tpc"]), min(self.channel_map["stpc"])),
            max(max(self.channel_map["tpc"]), max(self.channel_map["stpc"])),
        )

    def compute(self, records, start, end):
        # FIXME: This is going to make the same lone_hit having different record_i,
        # FIXME: between in salt mode and others
        # FIXME: surgery here; channel specification related
        # Based on saltax_mode, determine what channels to involve
        if self.saltax_mode == "salt":
            records = records[
                (records["channel"] >= self.schannel_starts_at)
                | (records["channel"] < self.n_tpc_pmts)
            ]
        elif self.saltax_mode == "simu":
            records = records[(records["channel"] >= self.schannel_starts_at)]
        elif self.saltax_mode == "data":
            records = records[(records["channel"] < self.n_tpc_pmts)]
        else:
            raise ValueError(f"Unknown saltax_mode {self.saltax_mode}")

        result = self._compute(records, start, end)

        # FIXME: surgery here; shifted lone_hits' channel for those which were salted
        mask_salted_lone_hits = result["lone_hits"]["channel"] >= self.schannel_starts_at
        result["lone_hits"]["channel"][mask_salted_lone_hits] -= self.schannel_starts_at
        # Sanity check on channels non-negative
        assert np.all(result["lone_hits"]["channel"] >= 0), "Negative channel number in lone_hits"
        # Sanity check that no lone_hits are in peaklets
        is_still_lone_hit = strax.fully_contained_in(result["lone_hits"], result["peaklets"]) == -1
        assert np.all(is_still_lone_hit), "Some lone_hits are in peaklets!?"

        return result


src = inspect.getsource(Peaklets.compute)
olds = [
    """
        peaklets = strax.find_peaks(
""",
    """
        strax.sum_waveform(
            peaklets, hitlets, r, rlinks, self.to_pe, n_top_channels=n_top_pmts_if_digitize_top
        )
""",
]
news = [
    """
        peaklets = find_peaks(
""",
    """
        sum_waveform(
            peaklets, hitlets, r, rlinks, self.to_pe, n_top_channels=n_top_pmts_if_digitize_top
        )
""",
]
src = replace_source(src, olds, news)
src = textwrap.dedent(src)
exec(src, globals())
SPeaklets._compute = compute  # type: ignore[name-defined]


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
    """
    strax.sum_waveform(peaks, hitlets, records, rlinks, to_pe, n_top_channels, peak_list)
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
            if ch >= SCHANNEL_STARTS_AT:
                ch_shifted = ch - SCHANNEL_STARTS_AT
            else:
                ch_shifted = ch

            if channel_saturated[ch]:
                b_pulse[ch_shifted, slice(*b_slice)] += r["data"][slice(*r_slice)]
                b_index[ch_shifted, np.argmin(b_index[ch_shifted])] = record_i
            else:
                b_sumwf[slice(*b_slice)] += r["data"][slice(*r_slice)] * to_pe[ch]
""",
    """
    sum_waveform(peaks, hitlets, records, rlinks, to_pe, n_top_channels, peak_list)
""",
]
src = replace_source(src, olds, news)
exec(src)


src = inspect.getsource(strax.find_peaks)
olds = [
    "cache=True",
    """
        area_per_channel[hit["channel"]] += hit_area_pe
""",
]
news = [
    "cache=False",
    """
        # Manually shift channels for area_per_channel
        if hit["channel"] >= SCHANNEL_STARTS_AT:
            area_per_channel[hit["channel"] - SCHANNEL_STARTS_AT] += hit_area_pe
        else:
            area_per_channel[hit["channel"]] += hit_area_pe
""",
]
src = replace_source(src, olds, news)
exec(src)


# FIXME: surgery here; top/bot array related
src = inspect.getsource(strax.sum_waveform)
olds = [
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
    "cache=False",
    """
            # Shift salted channel
            ch = h["channel"]
            ch_shifted = ch
            if ch >= n_channels:
                ch_shifted = ch - SCHANNEL_STARTS_AT
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
# this assignment is needed because `PeakSplitter.__call__` calls `strax.sum_waveform`
setattr(strax, "sum_waveform", sum_waveform)  # type: ignore[name-defined]
