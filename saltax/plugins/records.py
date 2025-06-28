from straxen.plugins.records.records import count_pulses
import numpy as np
import strax
import straxen
from straxen.plugins.records.records import NO_PULSE_COUNTS

SCHANNEL_STARTS_AT = 3000
export, __all__ = strax.exporter()
__all__.extend(["NO_PULSE_COUNTS"])


@export
class SPulseProcessing(straxen.PulseProcessing):
    """
    Split raw_records into:
     - (tpc) records
     - aqmon_records
     - pulse_counts

    For TPC records, apply basic processing:
        1. Flip, baseline, and integrate the waveform
        2. Apply software HE veto after high-energy peaks.
        3. Find hits, apply linear filter, and zero outside hits.

    For Simulated ones, repeat the same but also with channel number shifted.

    pulse_counts holds some average information for the individual PMT
    channels for each chunk of raw_records. This includes e.g.
    number of recorded pulses, lone_pulses (pulses which do not
    overlap with any other pulse), or mean values of baseline and
    baseline rms channel.
    """

    __version__ = "0.0.2"
    depends_on = ("raw_records", "raw_records_simu")

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

        # Do not trust in DAQ + strax.baseline to leave the
        # out-of-bounds samples to zero.
        strax.zero_out_of_bounds(r)
        strax.zero_out_of_bounds(r_simu)

        strax.baseline(
            r,
            baseline_samples=self.baseline_samples,
            allow_sloppy_chunking=self.allow_sloppy_chunking,
            flip=True,
        )
        strax.baseline(
            r_simu,
            baseline_samples=self.baseline_samples,
            allow_sloppy_chunking=self.allow_sloppy_chunking,
            flip=True,
        )

        strax.integrate(r)
        strax.integrate(r_simu)

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

        if len(r):
            # Find hits
            # -- before filtering,since this messes with the with the S/N
            hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

            if self.pmt_pulse_filter:
                # Filter to concentrate the PMT pulses
                strax.filter_records(r, np.array(self.pmt_pulse_filter))

            le, re = self.save_outside_hits
            r = strax.cut_outside_hits(r, hits, left_extension=le, right_extension=re)

            # Probably overkill, but just to be sure...
            strax.zero_out_of_bounds(r)

        if len(r_simu):
            # Find hits
            # -- before filtering,since this messes with the with the S/N
            hits_simu = strax.find_hits(r_simu, min_amplitude=self.hit_thresholds)

            if self.pmt_pulse_filter:
                # Filter to concentrate the PMT pulses
                strax.filter_records(r_simu, np.array(self.pmt_pulse_filter))

            le, re = self.save_outside_hits
            r_simu = strax.cut_outside_hits(
                r_simu, hits_simu, left_extension=le, right_extension=re
            )

            # Probably overkill, but just to be sure...
            strax.zero_out_of_bounds(r_simu)

        # Shift the SALT channels
        r_simu = shift_salt_channels(r_simu)

        # Merge the simulated and real records
        r = np.concatenate((r, r_simu))  # time stamps are NOT sorted anymore

        return dict(records=r, pulse_counts=pulse_counts, veto_regions=veto_regions)


def shift_salt_channels(r, n_channel_shift=SCHANNEL_STARTS_AT):
    """Shifts the channel numbers of the SALT PMTs.

    :param r: records (from fuse)
    :param n_channel_shift: number of channels to shift, default is
        SCHANNEL_STARTS_AT
    :return: raw_records with shifted channel numbers
    """
    r["channel"] += n_channel_shift
    return r
