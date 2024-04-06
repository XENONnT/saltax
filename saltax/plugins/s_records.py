from immutabledict import immutabledict
from straxen.plugins.records.records import count_pulses
import numpy as np
import strax
import straxen
from straxen.plugins.records.records import NO_PULSE_COUNTS

SCHANNEL_STARTS_AT = 3000
export, __all__ = strax.exporter()
__all__ += ["NO_PULSE_COUNTS"]


@export
class SPulseProcessing(strax.Plugin):
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

    parallel = "process"
    rechunk_on_save = immutabledict(records=False, veto_regions=True, pulse_counts=True)
    compressor = "zstd"

    depends_on = ("raw_records", "raw_records_simu")

    provides = ("records", "veto_regions", "pulse_counts")
    data_kind = {k: k for k in provides}
    save_when = immutabledict(
        records=strax.SaveWhen.TARGET,
        veto_regions=strax.SaveWhen.TARGET,
        pulse_counts=strax.SaveWhen.ALWAYS,
    )

    hev_gain_model = straxen.URLConfig(
        default=None, infer_type=False, help="PMT gain model used in the software high-energy veto."
    )

    baseline_samples = straxen.URLConfig(
        default=40,
        infer_type=False,
        help="Number of samples to use at the start of the pulse to determine " "the baseline",
    )

    # Tail veto options
    tail_veto_threshold = straxen.URLConfig(
        default=0,
        infer_type=False,
        help=(
            "Minimum peakarea in PE to trigger tail veto."
            "Set to None, 0 or False to disable veto."
        ),
    )

    tail_veto_duration = straxen.URLConfig(
        default=int(3e6), infer_type=False, help="Time in ns to veto after large peaks"
    )

    tail_veto_resolution = straxen.URLConfig(
        default=int(1e3),
        infer_type=False,
        help="Time resolution in ns for pass-veto waveform summation",
    )

    tail_veto_pass_fraction = straxen.URLConfig(
        default=0.05, infer_type=False, help="Pass veto if maximum amplitude above max * fraction"
    )

    tail_veto_pass_extend = straxen.URLConfig(
        default=3,
        infer_type=False,
        help="Extend pass veto by this many samples (tail_veto_resolution!)",
    )

    max_veto_value = straxen.URLConfig(
        default=None,
        infer_type=False,
        help="Optionally pass a HE peak that exceeds this absolute area. "
        "(if performing a hard veto, can keep a few statistics.)",
    )

    # PMT pulse processing options
    pmt_pulse_filter = straxen.URLConfig(
        default=None, infer_type=False, help="Linear filter to apply to pulses, will be normalized."
    )

    save_outside_hits = straxen.URLConfig(
        default=(3, 20),
        infer_type=False,
        help="Save (left, right) samples besides hits; cut the rest",
    )

    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of TPC PMTs")

    check_raw_record_overlaps = straxen.URLConfig(
        default=True,
        track=False,
        infer_type=False,
        help="Crash if any of the pulses in raw_records overlap with others " "in the same channel",
    )

    allow_sloppy_chunking = straxen.URLConfig(
        default=False,
        track=False,
        infer_type=False,
        help=(
            "Use a default baseline for incorrectly chunked fragments. "
            "This is a kludge for improperly converted XENON1T data."
        ),
    )

    hit_min_amplitude = straxen.URLConfig(
        track=True,
        infer_type=False,
        default="cmt://hit_thresholds_tpc?version=ONLINE&run_id=plugin.run_id",
        help="Minimum hit amplitude in ADC counts above baseline. "
        "Specify as a tuple of length n_tpc_pmts, or a number,"
        'or a string like "pmt_commissioning_initial" which means calling'
        "hitfinder_thresholds.py"
        "or a tuple like (correction=str, version=str, nT=boolean),"
        "which means we are using cmt.",
    )

    def infer_dtype(self):
        # Get record_length from the plugin making raw_records
        self.record_length = strax.record_length_from_dtype(
            self.deps["raw_records"].dtype_for("raw_records")
        )

        dtype = dict()
        for p in self.provides:
            if "records" in p:
                dtype[p] = strax.record_dtype(self.record_length)
        dtype["veto_regions"] = strax.hit_dtype
        dtype["pulse_counts"] = straxen.pulse_count_dtype(self.n_tpc_pmts)

        return dtype

    def setup(self):
        self.hev_enabled = self.hev_gain_model is not None and self.tail_veto_threshold
        if self.hev_enabled:
            self.to_pe = self.hev_gain_model
        self.hit_thresholds = self.hit_min_amplitude

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

    :param r: records (from wfsim)
    :param n_channel_shift: number of channels to shift, default is
        SCHANNEL_STARTS_AT
    :return: raw_records with shifted channel numbers
    """
    r["channel"] += n_channel_shift
    return r
