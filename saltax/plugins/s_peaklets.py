import numba
import numpy as np
import strax
from strax.processing.peak_building import _build_hit_waveform
from strax import utils
from immutabledict import immutabledict
from strax.processing.general import _touching_windows
from strax.dtypes import DIGITAL_SUM_WAVEFORM_CHANNEL
from .s_records import SCHANNEL_STARTS_AT
import straxen
from straxen.plugins.peaklets.peaklets import hit_max_sample, get_tight_coin, drop_data_top_field


export, __all__ = strax.exporter()

@export
class SPeaklets(strax.Plugin):
    """
    Split records into:
     - peaklets
     - lone_hits

    Peaklets are very aggressively split peaks such that we are able
    to find S1-S2s even if they are close to each other. (S2) Peaks
    that are split into too many peaklets will be merged later on.

    To get Peaklets from records apply/do:
        1. Hit finding
        2. Peak finding
        3. Peak splitting using the natural breaks algorithm
        4. Compute the digital sum waveform

    Lone hits are all hits which are outside of any peak. The area of
    lone_hits includes the left and right hit extension, except the
    extension overlaps with any peaks or other hits.
    """
    depends_on = ('records',)
    provides = ('peaklets', 'lone_hits')
    data_kind = dict(peaklets='peaklets',
                     lone_hits='lone_hits')
    parallel = 'process'
    compressor = 'zstd'

    __version__ = '0.0.0'

    peaklet_gap_threshold = straxen.URLConfig(
        default=700, infer_type=False,
        help="No hits for this many ns triggers a new peak")

    peak_left_extension = straxen.URLConfig(
        default=30, infer_type=False,
        help="Include this many ns left of hits in peaks")

    peak_right_extension = straxen.URLConfig(
        default=200, infer_type=False,
        help="Include this many ns right of hits in peaks")

    peak_min_pmts = straxen.URLConfig(
        default=2, infer_type=False,
        help="Minimum number of contributing PMTs needed to define a peak")

    peak_split_gof_threshold = straxen.URLConfig(
        # See https://xe1t-wiki.lngs.infn.it/doku.php?id=
        # xenon:xenonnt:analysis:strax_clustering_classification
        # #natural_breaks_splitting
        # for more information
        default=(
            None,  # Reserved
            ((0.5, 1.0), (6.0, 0.4)),
            ((2.5, 1.0), (5.625, 0.4))), infer_type=False,
        help='Natural breaks goodness of fit/split threshold to split '
             'a peak. Specify as tuples of (log10(area), threshold).')

    peak_split_filter_wing_width = straxen.URLConfig(
        default=70, infer_type=False,
        help='Wing width of moving average filter for '
             'low-split natural breaks')

    peak_split_min_area = straxen.URLConfig(
        default=40., infer_type=False,
        help='Minimum area to evaluate natural breaks criterion. '
             'Smaller peaks are not split.')

    peak_split_iterations = straxen.URLConfig(
        default=20, infer_type=False,
        help='Maximum number of recursive peak splits to do.')

    diagnose_sorting = straxen.URLConfig(
        track=False, default=False, infer_type=False,
        help="Enable runtime checks for sorting and disjointness")

    gain_model = straxen.URLConfig(
        infer_type=False,
        help='PMT gain model. Specify as URL or explicit value'
    )

    gain_model_mc = straxen.URLConfig(
        infer_type=False,
        help='PMT gain model. Specify as URL or explicit value'
    )

    tight_coincidence_window_left = straxen.URLConfig(
        default=50, infer_type=False,
        help="Time range left of peak center to call a hit a tight coincidence (ns)")

    tight_coincidence_window_right = straxen.URLConfig(
        default=50, infer_type=False,
        help="Time range right of peak center to call a hit a tight coincidence (ns)")

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help='Number of TPC PMTs')

    n_top_pmts = straxen.URLConfig(
        type=int,
        help="Number of top TPC array PMTs")

    sum_waveform_top_array = straxen.URLConfig(
        default=True,
        type=bool,
        help='Digitize the sum waveform of the top array separately'
    )

    saturation_correction_on = straxen.URLConfig(
        default=True, infer_type=False,
        help='On off switch for saturation correction')

    saturation_reference_length = straxen.URLConfig(
        default=100, infer_type=False,
        help="Maximum number of reference sample used "
             "to correct saturated samples")

    saturation_min_reference_length = straxen.URLConfig(
        default=20, infer_type=False,
        help="Minimum number of reference sample used "
             "to correct saturated samples")

    peaklet_max_duration = straxen.URLConfig(
        default=int(10e6), infer_type=False,
        help="Maximum duration [ns] of a peaklet")

    channel_map = straxen.URLConfig(
        track=False, type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) "
             "channel number.")

    hit_min_amplitude = straxen.URLConfig(
        track=True, infer_type=False,
        default='cmt://hit_thresholds_tpc?version=ONLINE&run_id=plugin.run_id',
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_tpc_pmts, or a number,'
             'or a string like "pmt_commissioning_initial" which means calling'
             'hitfinder_thresholds.py'
             'or a tuple like (correction=str, version=str, nT=boolean),'
             'which means we are using cmt.'
    )

    def infer_dtype(self):
        return dict(
            peaklets=strax.peak_dtype(
                n_channels=self.n_tpc_pmts,
                digitize_top=self.sum_waveform_top_array,
            ),
            lone_hits=strax.hit_dtype,
        )
    
    def setup(self):
        if self.peak_min_pmts > 2:
            # Can fix by re-splitting,
            raise NotImplementedError(
                f"Raising the peak_min_pmts to {self.peak_min_pmts} "
                f"interferes with lone_hit definition. "
                f"See github.com/XENONnT/straxen/issues/295")

        # Get the gain model
        # It should have length 3494 for XENONnT
        to_pe = np.zeros(SCHANNEL_STARTS_AT + self.n_tpc_pmts, dtype=np.float64)
        self.to_pe_data = self.gain_model
        self.to_pe_simu = self.gain_model_mc
        to_pe[:self.n_tpc_pmts] = self.to_pe_data
        to_pe[SCHANNEL_STARTS_AT:] = self.to_pe_simu
        self.to_pe = to_pe

        # Get the hitfinder thresholds
        hit_thresholds = np.zeros(SCHANNEL_STARTS_AT + self.n_tpc_pmts, dtype=np.int64)
        hit_thresholds[:self.n_tpc_pmts] = self.hit_min_amplitude
        hit_thresholds[SCHANNEL_STARTS_AT:] = self.hit_min_amplitude
        self.hit_thresholds = hit_thresholds

        self.channel_range = self.channel_map['tpc']

        # Override strax.sum_waveform
        setattr(strax, "sum_waveform", sum_waveform_salted)
        
    def compute(self, records, start, end):
        # Throw away any non-TPC records
        r = records[(records['channel']>=SCHANNEL_STARTS_AT)|
                    (records['channel']<self.n_tpc_pmts)]

        # 988 channels
        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

        # Remove hits in zero-gain channels
        # they should not affect the clustering!
        hits = hits[self.to_pe[hits['channel']] != 0]

        hits = strax.sort_by_time(hits)

        # FIXME: surgery here; top/bot array related
        # Use peaklet gap threshold for initial clustering
        # based on gaps between hits
        peaklets = find_peaks(
            hits, self.to_pe,
            gap_threshold=self.peaklet_gap_threshold,
            left_extension=self.peak_left_extension,
            right_extension=self.peak_right_extension,
            min_channels=self.peak_min_pmts,
            # NB, need to have the data_top field here, will discard if not digitized later
            result_dtype=strax.peak_dtype(n_channels=self.n_tpc_pmts, digitize_top=True),
            max_duration=self.peaklet_max_duration,
        )

        # Make sure peaklets don't extend out of the chunk boundary
        # This should be very rare in normal data due to the ADC pretrigger
        # window.
        self.clip_peaklet_times(peaklets, start, end)

        # Get hits outside peaklets, and store them separately.
        # fully_contained is OK provided gap_threshold > extension,
        # which is asserted inside strax.find_peaks.
        is_lone_hit = strax.fully_contained_in(hits, peaklets) == -1
        lone_hits = hits[is_lone_hit]

        # Update the area of lone_hits to the integral in ADCcounts x samples
        strax.integrate_lone_hits(
            lone_hits, records, peaklets,
            save_outside_hits=(self.peak_left_extension,
                               self.peak_right_extension),
            n_channels=len(self.to_pe))
        
        # Compute basic peak properties -- needed before natural breaks
        hits = hits[~is_lone_hit]
        # Define regions outside of peaks such that _find_hit_integration_bounds
        # is not extended beyond a peak.
        outside_peaks = self.create_outside_peaks_region(peaklets, start, end)
        # Still assuming we have 2*n_tpc_channels to reduce bias from pileup cases
        strax.find_hit_integration_bounds(
            hits, outside_peaks, records,
            save_outside_hits=(self.peak_left_extension,
                               self.peak_right_extension),
            n_channels=len(self.to_pe),
            allow_bounds_beyond_records=True,
        )

        # Transform hits to hitlets for naming conventions. A hit refers
        # to the central part above threshold a hitlet to the entire signal
        # including the left and right extension.
        # (We are not going to use the actual hitlet data_type here.)
        hitlets = hits
        del hits

        hitlets['time'] -= (hitlets['left'] - hitlets['left_integration']) * hitlets['dt']
        hitlets['length'] = hitlets['right_integration'] - hitlets['left_integration']
        hitlets = strax.sort_by_time(hitlets)
        rlinks = strax.record_links(records)

        # If sum_waveform_top_array is false, don't digitize the top array
        n_top_pmts_if_digitize_top = self.n_top_pmts if self.sum_waveform_top_array else -1
        # FIXME: surgery here; top/bot array related
        strax.sum_waveform(peaklets, hitlets, r, rlinks, self.to_pe, n_top_channels=n_top_pmts_if_digitize_top)

        strax.compute_widths(peaklets)

        # hitlets here are still 3494 channels
        # Split peaks using low-split natural breaks;
        # see https://github.com/XENONnT/straxen/pull/45
        # and https://github.com/AxFoundation/strax/pull/225
        peaklets = strax.split_peaks(
            peaklets, hitlets, r, rlinks, self.to_pe,
            algorithm='natural_breaks',
            threshold=self.natural_breaks_threshold,
            split_low=True,
            filter_wing_width=self.peak_split_filter_wing_width,
            min_area=self.peak_split_min_area,
            do_iterations=self.peak_split_iterations,
            n_top_channels=n_top_pmts_if_digitize_top
        )

        # Saturation correction using non-saturated channels
        # similar method used in pax
        # see https://github.com/XENON1T/pax/pull/712
        # Cases when records is not writeable for unclear reason
        # only see this when loading 1T test data
        # more details on https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
        if not r['data'].flags.writeable:
            r = r.copy()

        if self.saturation_correction_on:
            peak_list = peak_saturation_correction(
                r, rlinks, peaklets, hitlets, self.to_pe,
                reference_length=self.saturation_reference_length,
                min_reference_length=self.saturation_min_reference_length,
                n_top_channels=n_top_pmts_if_digitize_top,
            )

            # Compute the width again for corrected peaks
            strax.compute_widths(peaklets, select_peaks_indices=peak_list)

        hitlet_time_shift = (hitlets['left'] - hitlets['left_integration']) * hitlets['dt']
        hit_max_times = hitlets['time'] + hitlet_time_shift  # add time shift again to get correct maximum
        hit_max_times += hitlets['dt'] * hit_max_sample(records, hitlets)

        # Compute tight coincidence level.
        # Making this a separate plugin would
        # (a) doing hitfinding yet again (or storing hits)
        # (b) increase strax memory usage / max_messages,
        #     possibly due to its currently primitive scheduling.
        hit_max_times_argsort = np.argsort(hit_max_times)
        sorted_hit_max_times = hit_max_times[hit_max_times_argsort]
        sorted_hit_channels = hitlets['channel'][hit_max_times_argsort]
        peaklet_max_times = (
            peaklets['time']
            + np.argmax(peaklets['data'], axis=1) * peaklets['dt'])
        peaklets['tight_coincidence'] = get_tight_coin(
            sorted_hit_max_times,
            sorted_hit_channels,
            peaklet_max_times,
            self.tight_coincidence_window_left,
            self.tight_coincidence_window_right,
            self.channel_range)

        # Add max and min time difference between apexes of hits
        self.add_hit_features(hitlets, hit_max_times, peaklets)

        if self.diagnose_sorting and len(r):
            assert np.diff(r['time']).min(initial=1) >= 0, "Records not sorted"
            assert np.diff(hitlets['time']).min(initial=1) >= 0, "Hits/Hitlets not sorted"
            assert np.all(peaklets['time'][1:]
                          >= strax.endtime(peaklets)[:-1]), "Peaks not disjoint"

        # Update nhits of peaklets:
        counts = strax.touching_windows(hitlets, peaklets)
        counts = np.diff(counts, axis=1).flatten()
        peaklets['n_hits'] = counts

        # Drop the data_top field
        if n_top_pmts_if_digitize_top <= 0:
            peaklets = drop_data_top_field(peaklets, self.dtype_for('peaklets'))

        # Check channel of peaklets
        peaklets_unique_channel = np.unique(peaklets['channel'])
        if (peaklets_unique_channel == DIGITAL_SUM_WAVEFORM_CHANNEL).sum() > 1:
            raise ValueError(
                f'Found channel number of peaklets other than {DIGITAL_SUM_WAVEFORM_CHANNEL}')
        # Check tight_coincidence
        if not np.all(peaklets['n_hits'] >= peaklets['tight_coincidence']):
            raise ValueError(
                f'Found n_hits less than tight_coincidence')

        return dict(peaklets=peaklets,
                    lone_hits=lone_hits)

    def natural_breaks_threshold(self, peaks):
        """
        Pasted from https://github.com/XENONnT/straxen/blob/5f232eb2c1ab39e11fb14d4e6ee2db369ed2c2ec/straxen/plugins/peaklets/peaklets.py#L332-L348
        """
        rise_time = -peaks['area_decile_from_midpoint'][:, 1]

        # This is ~1 for an clean S2, ~0 for a clean S1,
        # and transitions gradually in between.
        f_s2 = 8 * np.log10(rise_time.clip(1, 1e5) / 100)
        f_s2 = 1 / (1 + np.exp(-f_s2))

        log_area = np.log10(peaks['area'].clip(1, 1e7))
        thresholds = self.peak_split_gof_threshold
        return (
                f_s2 * np.interp(
            log_area,
            *np.transpose(thresholds[2]))
                + (1 - f_s2) * np.interp(
            log_area,
            *np.transpose(thresholds[1])))

    @staticmethod
    def clip_peaklet_times(peaklets, start, end):
        straxen.plugins.peaklets.Peaklets.clip_peaklet_times(peaklets, 
                                                             start, 
                                                             end)

    @staticmethod
    def create_outside_peaks_region(peaklets, start, end):
        """
        Creates time intervals which are outside peaks.

        :param peaklets: Peaklets for which intervals should be computed.
        :param start: Chunk start
        :param end: Chunk end
        :return: array of strax.time_fields dtype.
        """
        outside_peaks = straxen.plugins.peaklets.Peaklets.create_outside_peaks_region(peaklets, 
                                                                                      start, 
                                                                                      end)
        return outside_peaks

    @staticmethod
    def add_hit_features(hitlets, hit_max_times, peaklets):
        """
        Create hits timing features
        :param hitlets_max: hitlets with only max height time.
        :param peaklets: Peaklets for which intervals should be computed.
        :return: array of peaklet_timing dtype.
        """
        straxen.plugins.peaklets.peaklets.Peaklets.add_hit_features(hitlets, 
                                                                    hit_max_times, 
                                                                    peaklets)


@numba.jit(nopython=True, nogil=True, cache=False)
def peak_saturation_correction(records, rlinks, peaks, hitlets, to_pe,
                               reference_length=100,
                               min_reference_length=20,
                               use_classification=False,
                               n_top_channels=0,
                               ):
    """WARNING: This probably doesn't work when we have the salted channel also saturated!!!
    We will be using only the real TPC channels to correct the saturation!!! This is dangerous
    if you are salting things outside WIMP/LowER regions!!!
    Correct the area and per pmt area of peaks from saturation
    :param records: Records
    :param rlinks: strax.record_links of corresponding records.
    :param peaks: Peaklets / Peaks
    :param hitlets: Hitlets found in records to build peaks.
        (Hitlets are hits including the left/right extension)
    :param to_pe: adc to PE conversion (length should equal number of PMTs)
    :param reference_length: Maximum number of reference sample used
    to correct saturated samples
    :param min_reference_length: Minimum number of reference sample used
    to correct saturated samples
    :param use_classification: Option of using classification to pick only S2
    :param n_top_channels: Number of top array channels.
    """

    if not len(records):
        return
    if not len(peaks):
        return

    # Search for peaks with saturated channels
    mask = peaks['n_saturated_channels'] > 0
    if use_classification:
        mask &= peaks['type'] == 2
    peak_list = np.where(mask)[0]
    # Look up records that touch each peak
    record_ranges = _touching_windows(
        records['time'],
        strax.endtime(records),
        peaks[peak_list]['time'],
        strax.endtime(peaks[peak_list]))

    # Create temporary arrays for calculation
    dt = records[0]['dt']
    n_channels = len(peaks[0]['saturated_channel'])
    len_buffer = np.max(peaks['length'] * peaks['dt']) // dt + 1
    max_nrecord = len_buffer // len(records[0]['data']) + 1

    # Buff the sum wf [pe] of non-saturated channels
    b_sumwf = np.zeros(len_buffer, dtype=np.float32)
    # Buff the records 'data' [ADC] in saturated channels
    b_pulse = np.zeros((n_channels, len_buffer), dtype=np.int16)
    # Buff the corresponding record index of saturated channels
    b_index = np.zeros((n_channels, max_nrecord), dtype=np.int64)

    # Main
    for ix, peak_i in enumerate(peak_list):
        # reset buffers
        b_sumwf[:] = 0
        b_pulse[:] = 0
        b_index[:] = -1

        p = peaks[peak_i]
        channel_saturated = p['saturated_channel'] > 0

        for record_i in range(record_ranges[ix][0], record_ranges[ix][1]):
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r['time'] // dt, r['length'],
                p['time'] // dt, p['length'] * p['dt'] // dt)

            # Shift channels to handle salted channels
            ch = r['channel']
            if ch >= SCHANNEL_STARTS_AT:
                ch_shifted = ch - SCHANNEL_STARTS_AT
            else:
                ch_shifted = ch

            if channel_saturated[ch]:
                b_pulse[ch_shifted, slice(*b_slice)] += r['data'][slice(*r_slice)]
                b_index[ch_shifted, np.argmin(b_index[ch_shifted])] = record_i
            else:
                b_sumwf[slice(*b_slice)] += r['data'][slice(*r_slice)] \
                                            * to_pe[ch]

        _peak_saturation_correction_inner(
            channel_saturated, records, p,
            to_pe, b_sumwf, b_pulse, b_index,
            reference_length, min_reference_length)

        # Back track sum wf downsampling
        peaks[peak_i]['length'] = p['length'] * p['dt'] / dt
        peaks[peak_i]['dt'] = dt

    strax.sum_waveform(peaks, hitlets, records, rlinks, to_pe, n_top_channels, peak_list)
    return peak_list


@numba.jit(nopython=True, nogil=True, cache=True)
def _peak_saturation_correction_inner(channel_saturated, records, p,
                                      to_pe, b_sumwf, b_pulse, b_index,
                                      reference_length=100,
                                      min_reference_length=20,
                                      ):
    """WARNING: This probably doesn't work when we have the salted channel also saturated!!!
    We will be using only the real TPC channels to correct the saturation!!! This is dangerous
    if you are salting things outside WIMP/LowER regions!!!
    Would add a third level loop in peak_saturation_correction
    Which is not ideal for numba, thus this function is written
    :param channel_saturated: (bool, n_channels)
    :param p: One peak/peaklet
    :param to_pe: adc to PE conversion (length should equal number of PMTs)
    :param b_sumwf, b_pulse, b_index: Filled buffers
    """
    dt = records['dt'][0]
    n_channels = len(channel_saturated)

    for ch in range(n_channels):
        if not channel_saturated[ch]:
            continue
        b = b_pulse[ch]
        r0 = records[b_index[ch][0]]

        # Define the reference region as reference_length before the first saturation point
        # unless there are not enough samples
        bl = np.inf
        for record_i in b_index[ch]:
            if record_i == -1:
                break
            bl = min(bl, records['baseline'][record_i])

        s0 = np.argmax(b >= np.int16(bl))
        ref = slice(max(0, s0 - reference_length), s0)

        if (b[ref] * to_pe[ch] > 1).sum() < min_reference_length:
            # the pulse is saturated, but there are not enough reference samples to get a good ratio
            # This actually distinguished between S1 and S2 and will only correct S2 signals
            continue
        if (b_sumwf[ref] > 1).sum() < min_reference_length:
            # the same condition applies to the waveform model
            continue
        if np.sum(b[ref]) * to_pe[ch] / np.sum(b_sumwf[ref]) > 1:
            # The pulse is saturated, but insufficient information is available in the other channels
            # to reliably reconstruct it
            continue

        scale = np.sum(b[ref]) / np.sum(b_sumwf[ref])

        # Loop over the record indices of the saturated channel (saved in b_index buffer)
        for record_i in b_index[ch]:
            if record_i == -1:
                break
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r['time'] // dt, r['length'],
                p['time'] // dt + s0, p['length'] * p['dt'] // dt - s0)

            if r_slice[1] == r_slice[0]:  # This record proceeds saturation
                continue
            b_slice = b_slice[0] + s0, b_slice[1] + s0

            # First is finding the highest point in the desaturated record
            # because we need to bit shift the whole record if it exceeds int16 range
            apax = scale * max(b_sumwf[slice(*b_slice)])

            if np.int32(apax) >= 2 ** 15:  # int16(2**15) is -2**15
                bshift = int(np.floor(np.log2(apax) - 14))

                tmp = r['data'].astype(np.int32)
                tmp[slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale

                r['area'] = np.sum(tmp)  # Auto covert to int64
                r['data'][:] = np.right_shift(tmp, bshift)
                r['amplitude_bit_shift'] += bshift
            else:
                r['data'][slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale
                r['area'] = np.sum(r['data'])


@export
@utils.growing_result(dtype=strax.dtypes.peak_dtype(), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def find_peaks(hits, adc_to_pe,
               gap_threshold=300,
               left_extension=20, right_extension=150,
               min_area=0,
               min_channels=2,
               max_duration=10_000_000,
               _result_buffer=None, result_dtype=None):
    """Return peaks made from grouping hits together. Modified parts related to area_per_channel
    Assumes all hits have the same dt
    :param hits: Hit (or any interval) to group
    :param left_extension: Extend peaks by this many ns left
    :param right_extension: Extend peaks by this many ns right
    :param gap_threshold: No hits for this much ns means new peak
    :param min_area: Peaks with less than min_area are not returned
    :param min_channels: Peaks with less contributing channels are not returned
    :param max_duration: max duration time of merged peak in ns

    Modified based on https://github.com/AxFoundation/strax/blob/9b508f7f8d441bf1fe441695115d292c59ce631a/strax/processing/peak_building.py#L13
    """
    buffer = _result_buffer
    offset = 0
    if not len(hits):
        return
    assert hits[0]['dt'] > 0, "Hit does not indicate sampling time"
    assert min_channels >= 1, "min_channels must be >= 1"
    assert gap_threshold > left_extension + right_extension, \
        "gap_threshold must be larger than left + right extension"
    assert max(hits['channel']) < len(adc_to_pe), "more channels than to_pe"
    # Magic number comes from
    #   np.iinfo(p['dt'].dtype).max*np.shape(p['data'])[1] = 429496729400 ns
    # but numba does not like it
    assert left_extension+max_duration+right_extension < 429496729400, (
        "Too large max duration causes integer overflow")

    n_channels = len(buffer[0]['area_per_channel'])
    area_per_channel = np.zeros(n_channels, dtype=np.float32)

    in_peak = False
    peak_endtime = 0
    for hit_i, hit in enumerate(hits):
        p = buffer[offset]
        t0 = hit['time']
        dt = hit['dt']
        t1 = hit['time'] + dt * hit['length']

        if in_peak:
            # This hit continues an existing peak
            p['max_gap'] = max(p['max_gap'], t0 - peak_endtime)

        else:
            # This hit starts a new peak candidate
            area_per_channel *= 0
            peak_endtime = t1
            p['time'] = t0 - left_extension
            p['channel'] = DIGITAL_SUM_WAVEFORM_CHANNEL
            p['dt'] = dt
            # These are necessary as prev peak may have been rejected:
            p['n_hits'] = 0
            p['area'] = 0
            in_peak = True
            p['max_gap'] = 0

        # Add hit's properties to the current peak candidate

        # NB! One pulse can result in two hits, if it occours at the 
        # boundary of a record. This is the default of strax.find_hits.
        p['n_hits'] += 1

        peak_endtime = max(peak_endtime, t1)
        hit_area_pe = hit['area'] * adc_to_pe[hit['channel']]
        
        # Manually shift channels for area_per_channel
        if hit['channel']>=SCHANNEL_STARTS_AT:
            area_per_channel[hit['channel']-SCHANNEL_STARTS_AT] += hit_area_pe
        else:
            area_per_channel[hit['channel']] += hit_area_pe
        p['area'] += hit_area_pe

        # Look at the next hit to see if THIS hit is the last in a peak.
        # If this is the final hit, it is last by definition.
        # Finally, make sure that if we include the next hit, we are not
        # exceeding the max_duration.
        is_last_hit = hit_i == len(hits) - 1
        peak_too_long = next_hit_is_far = False
        if not is_last_hit:
            # These can only be computed if there is a next hit
            next_hit = hits[hit_i + 1]
            next_hit_is_far = next_hit['time'] - peak_endtime >= gap_threshold
            # Peaks may not extend the max_duration
            peak_too_long = (next_hit['time'] - p['time']
                             + next_hit['dt'] * next_hit['length']
                             + left_extension
                             + right_extension) > max_duration
        if is_last_hit or next_hit_is_far or peak_too_long:
            # Next hit (if it exists) will initialize the new peak candidate
            in_peak = False

            # Do not save if tests are not met. Next hit will erase temp info
            if p['area'] < min_area:
                continue
            n_channels = (area_per_channel != 0).sum()
            if n_channels < min_channels:
                continue

            # Compute final quantities
            p['length'] = (peak_endtime - p['time'] + right_extension) / dt
            if p['length'] <= 0:
                # This is most likely caused by a negative dt
                raise ValueError(
                    "Caught attempt to save nonpositive peak length?!")
            p['area_per_channel'][:] = area_per_channel

            # Save the current peak, advance the buffer
            offset += 1
            if offset == len(buffer):
                yield offset
                offset = 0

    yield offset

@export
@numba.jit(nopython=True, nogil=True, cache=True)
def sum_waveform_salted(peaks, hits, records, record_links, adc_to_pe, n_top_channels=0,
                 select_peaks_indices=None):
    """Modified to handle array channels range.
    Compute sum waveforms for all peaks in peaks. Only builds summed
    waveform other regions in which hits were found. This is required
    to avoid any bias due to zero-padding and baselining.
    Will downsample sum waveforms if they do not fit in per-peak buffer

    :param peaks: Peaks for which the summed waveform should be build.
    :param hits: Hits which are inside peaks. Must be sorted according
        to record_i.
    :param records: Records to be used to build peaks.
    :param record_links: Tuple of previous and next records.
    :param n_top_channels: Number of top array channels.
    :param select_peaks_indices: Indices of the peaks for partial
    processing. In the form of np.array([np.int, np.int, ..]). If
    None (default), all the peaks are used for the summation.

    Assumes all peaks AND pulses have the same dt!
    """
    if not len(records):
        return
    if not len(peaks):
        return
    if select_peaks_indices is None:
        select_peaks_indices = np.arange(len(peaks))
    if not len(select_peaks_indices):
        return
    dt = records[0]['dt']
    n_samples_record = len(records[0]['data'])
    prev_record_i, next_record_i = record_links

    # Big buffer to hold even largest sum waveforms
    # Need a little more even for downsampling..
    swv_buffer = np.zeros(peaks['length'].max() * 2, dtype=np.float32)

    if n_top_channels > 0:
        twv_buffer = np.zeros(peaks['length'].max() * 2, dtype=np.float32)

    n_channels = len(peaks[0]['area_per_channel'])
    area_per_channel = np.zeros(n_channels, dtype=np.float32)

    # Hit index for hits in peaks
    left_h_i = 0
    # Create hit waveform buffer
    hit_waveform = np.zeros(hits['length'].max(), dtype=np.float32)

    for peak_i in select_peaks_indices:
        p = peaks[peak_i]
        # Clear the relevant part of the swv buffer for use
        # (we clear a bit extra for use in downsampling)
        p_length = p['length']
        swv_buffer[:min(2 * p_length, len(swv_buffer))] = 0

        if n_top_channels > 0:
            twv_buffer[:min(2 * p_length, len(twv_buffer))] = 0

        # Clear area and area per channel
        # (in case find_peaks already populated them)
        area_per_channel *= 0
        p['area'] = 0

        # Find first hit that contributes to this peak
        for left_h_i in range(left_h_i, len(hits)):
            h = hits[left_h_i]
            # TODO: need test that fails if we replace < with <= here
            if p['time'] < h['time'] + h['length'] * dt:
                break
        else:
            # Hits exhausted before peaks exhausted
            # TODO: this is a strange case, maybe raise warning/error?
            break

        # Scan over hits that overlap with peak
        for right_h_i in range(left_h_i, len(hits)):
            h = hits[right_h_i]
            record_i = h['record_i']
            
            # Shift salted channel
            ch = h['channel']
            ch_shifted = ch
            if ch >= n_channels:
                ch_shifted = ch - SCHANNEL_STARTS_AT

            assert p['dt'] == h['dt'], "Hits and peaks must have same dt"

            shift = (p['time'] - h['time']) // dt
            n_samples_hit = h['length']
            n_samples_peak = p_length

            if shift <= -n_samples_peak:
                # Hit is completely to the right of the peak;
                # we've seen all overlapping records
                break

            if n_samples_hit <= shift:
                # The (real) data in this record does not actually overlap
                # with the peak
                # (although a previous, longer hit did overlap)
                continue

            # Get overlapping samples between hit and peak:
            (h_start, h_end), (p_start, p_end) = strax.overlap_indices(
                h['time'] // dt, n_samples_hit,
                p['time'] // dt, n_samples_peak)

            hit_waveform[:] = 0

            # Get record which belongs to main part of hit (wo integration bounds):
            r = records[record_i]

            is_saturated = _build_hit_waveform(h, r, hit_waveform)

            # Now check if we also have to go to prev/next record due to integration bounds.
            # If bounds are outside of peak we chop when building the summed waveform later.
            if h['left_integration'] < 0 and prev_record_i[record_i] != -1:
                r = records[prev_record_i[record_i]]
                is_saturated |= _build_hit_waveform(h, r, hit_waveform)

            if h['right_integration'] > n_samples_record and next_record_i[record_i] != -1:
                r = records[next_record_i[record_i]]
                is_saturated |= _build_hit_waveform(h, r, hit_waveform)

            p['saturated_channel'][ch_shifted] |= is_saturated

            hit_data = hit_waveform[h_start:h_end]
            hit_data *= adc_to_pe[ch]
            swv_buffer[p_start:p_end] += hit_data

            if n_top_channels > 0:
                if ch_shifted < n_top_channels:
                    twv_buffer[p_start:p_end] += hit_data

            area_pe = hit_data.sum()
            area_per_channel[ch_shifted] += area_pe
            p['area'] += area_pe

        if n_top_channels > 0:
            strax.store_downsampled_waveform(p, swv_buffer, True, twv_buffer)
        else:
            strax.store_downsampled_waveform(p, swv_buffer)

        p['n_saturated_channels'] = p['saturated_channel'].sum()
        p['area_per_channel'][:] = area_per_channel

