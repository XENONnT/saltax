import numpy as np

import strax
import straxen

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
        default="salt", infer_type=False, help="'data', 'simu', or 'salt'"
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

        result = super().compute(records, start, end)

        # FIXME: surgery here; shifted lone_hits' channel for those which were salted
        mask_salted_lone_hits = result["lone_hits"]["channel"] >= self.schannel_starts_at
        result["lone_hits"]["channel"][mask_salted_lone_hits] -= self.schannel_starts_at

        # Sanity check on channels non-negative
        assert np.all(result["lone_hits"]["channel"] >= 0), "Negative channel number in lone_hits"
        # Sanity check that no lone_hits are in peaklets
        is_still_lone_hit = strax.fully_contained_in(result["lone_hits"], result["peaklets"]) == -1
        assert np.all(is_still_lone_hit), "Some lone_hits are in peaklets!?"

        return result
