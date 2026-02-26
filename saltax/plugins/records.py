import numpy as np
import strax
import straxen
from straxen.plugins.records.records import count_pulses

SCHANNEL_STARTS_AT = 3000

export, __all__ = strax.exporter()
__all__.extend(["SCHANNEL_STARTS_AT"])


@export
class SPulseProcessing(straxen.PulseProcessing):
    __version__ = "0.0.3"
    depends_on = ("raw_records", "raw_records_simu")

    schannel_starts_at = straxen.URLConfig(
        default=SCHANNEL_STARTS_AT,
        infer_type=False,
        help="Salting channel starts at this channel",
    )

    def compute(self, raw_records, raw_records_simu, start, end):
        if self.check_raw_record_overlaps:
            straxen.check_overlaps(raw_records, n_channels=3000)

        # Throw away any non-TPC records; this should only happen for XENON1T
        # converted data
        raw_records = raw_records[raw_records["channel"] < self.n_tpc_pmts]

        # Convert everything to the records data type -- adds extra fields.
        r = strax.raw_to_records(raw_records)
        r_simu = strax.raw_to_records(raw_records_simu)
        del raw_records, raw_records_simu

        for _r in (r, r_simu):
            # Do not trust in DAQ + strax.baseline to leave the
            # out-of-bounds samples to zero.
            strax.zero_out_of_bounds(_r)
            strax.baseline(
                _r,
                baseline_samples=self.baseline_samples,
                allow_sloppy_chunking=self.allow_sloppy_chunking,
                flip=True,
            )
            strax.integrate(_r)

        # Ignoring the ones from salt channels
        pulse_counts = count_pulses(r, self.n_tpc_pmts)
        pulse_counts["time"] = start
        pulse_counts["endtime"] = end

        # Ignoring the ones from salt channels
        if len(r) and self.hev_enabled:

            r, r_vetoed, veto_regions = straxen.software_he_veto(
                r,
                self.to_pe,
                end,
                area_threshold=self.tail_veto_threshold,
                veto_length=self.tail_veto_duration,
                veto_res=self.tail_veto_resolution,
                pass_veto_extend=self.tail_veto_pass_extend,
                pass_veto_fraction=self.tail_veto_pass_fraction,
                max_veto_value=self.max_veto_value,
            )

            # In the future, we'll probably want to sum the waveforms
            # inside the vetoed regions, so we can still save the "peaks".
            del r_vetoed

        else:
            veto_regions = np.zeros(0, dtype=strax.hit_dtype)

        for _r in (r, r_simu):
            if len(_r):
                # Find hits
                # -- before filtering,since this messes with the with the S/N
                hits = strax.find_hits(_r, min_amplitude=self.hit_thresholds)

                if self.pmt_pulse_filter:
                    # Filter to concentrate the PMT pulses
                    strax.filter_records(_r, np.array(self.pmt_pulse_filter))

                le, re = self.save_outside_hits
                _r = strax.cut_outside_hits(_r, hits, left_extension=le, right_extension=re)

                # Probably overkill, but just to be sure...
                strax.zero_out_of_bounds(_r)

        # Shift the SALT channels
        r_simu["channel"] += self.schannel_starts_at

        # Merge the simulated and real records
        r = np.concatenate((r, r_simu))
        r = strax.sort_by_time(r)

        return dict(records=r, pulse_counts=pulse_counts, veto_regions=veto_regions)
