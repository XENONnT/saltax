import strax
import numpy as np
import wfsim
import logging
from wfsim import extra_truth_dtype_per_pmt
from immutabledict import immutabledict
from wfsim.strax_interface import SimulatorPlugin, instruction_dtype, rand_instructions, instruction_from_csv

export, __all__ = strax.exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.interface')
log.setLevel('WARNING')


@export
class ChunkRawRecords(object):
    """
    Chunk the raw records from wfsim into strax chunks. Inherited from wfsim.ChunkRawRecords,
    with plugin names changed.
    """
    def __init__(self, config, rawdata_generator=wfsim.RawData, **kwargs):
        log.debug(f'Starting {self.__class__.__name__}')
        self.config = config
        log.debug(f'Setting raw data with {rawdata_generator.__name__}')
        self.rawdata = rawdata_generator(self.config, **kwargs)
        self.record_buffer = np.zeros(5000000, # 5e6 fragments of records
                                      dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH))
        truth_per_n_pmts = self._n_channels if config.get('per_pmt_truth') else False
        self.truth_dtype = extra_truth_dtype_per_pmt(truth_per_n_pmts)
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + self.truth_dtype + [('fill', bool)])

        self.blevel = 0  # buffer_filled_level

    def __call__(self, instructions, time_zero=None, **kwargs):
        """
        :param instructions: Structured array with instruction dtype in strax_interface module
        :param time_zero: Starting time of the first chunk, default: None
        """
        samples_per_record = strax.DEFAULT_RECORD_LENGTH
        if len(instructions) == 0: # Empty
            yield from np.array([], dtype=strax.raw_record_dtype(samples_per_record=samples_per_record))
            self.rawdata.source_finished = True
            return
        dt = self.config['sample_duration']
        buffer_length = len(self.record_buffer)
        rext = int(self.config['right_raw_extension'])
        cksz = int(self.config['chunk_size'] * 1e9)

        # Save the constants as privates
        # chunk_time_pre is the start of a chunk
        # chunk_time is the end of a chunk
        self.blevel = 0  # buffer_filled_level
        self.chunk_time_pre = time_zero - rext if time_zero else np.min(instructions['time']) - rext
        self.chunk_time = self.chunk_time_pre + cksz
        self.current_digitized_right = self.last_digitized_right = 0

        # Loop over the instructions, and fill the record buffer, and yield the results
        for channel, left, right, data in self.rawdata(instructions=instructions,
                                                       truth_buffer=self.truth_buffer,
                                                       **kwargs):
            pulse_length = right - left + 1
            # Number of records fragments needed to store the pulse
            records_needed = int(np.ceil(pulse_length / samples_per_record))

            # Update the digitized right
            if self.rawdata.right != self.current_digitized_right:
                self.last_digitized_right = self.current_digitized_right
                self.current_digitized_right = self.rawdata.right

            # Check if the buffer is full, if so, yield the results
            if self.rawdata.left * dt > self.chunk_time + rext:
                next_left_time = self.rawdata.left * dt
                log.debug(f'Pause sim loop at {self.chunk_time}, next pulse start at {next_left_time}')
                if (self.last_digitized_right + 1) * dt > self.chunk_time:
                    extend = (self.last_digitized_right + 1) * dt - self.chunk_time
                    self.chunk_time += extend 
                    log.debug(f'Chunk happenned during event, extending {extend} ns')
                yield from self.final_results()
                self.chunk_time_pre = self.chunk_time
                self.chunk_time += cksz

            if self.blevel + records_needed > buffer_length:
                log.warning('Chunck size too large, insufficient record buffer \n'
                            'No longer in sync if simulating nVeto with TPC \n'
                            'Consider reducing the chunk size')
                next_left_time = self.rawdata.left * dt
                self.chunk_time = (self.last_digitized_right + 1) * dt
                log.debug(f'Pause sim loop at {self.chunk_time}, next pulse start at {next_left_time}')
                yield from self.final_results()
                self.chunk_time_pre = self.chunk_time
                self.chunk_time += cksz

            if self.blevel + records_needed > buffer_length:
                log.warning('Pulse length too large, insufficient record buffer, skipping pulse')
                continue

            # WARNING baseline and area fields are zeros before finish_results
            s = slice(self.blevel, self.blevel + records_needed)
            self.record_buffer[s]['channel'] = channel
            self.record_buffer[s]['dt'] = dt
            self.record_buffer[s]['time'] = dt * (left + samples_per_record * np.arange(records_needed))
            self.record_buffer[s]['length'] = [min(pulse_length, samples_per_record * (i+1))
                                               - samples_per_record * i for i in range(records_needed)]
            self.record_buffer[s]['pulse_length'] = pulse_length
            self.record_buffer[s]['record_i'] = np.arange(records_needed)
            self.record_buffer[s]['data'] = np.pad(data,
                                                   (0, records_needed * samples_per_record - pulse_length),
                                                   'constant').reshape((-1, samples_per_record))
            self.blevel += records_needed

        self.last_digitized_right = self.current_digitized_right
        self.chunk_time = max((self.last_digitized_right + 1) * dt, self.chunk_time_pre + dt)
        yield from self.final_results()

    def final_results(self):
        """
        Yield the final results. 
        """
        records = self.record_buffer[:self.blevel]  # No copying the records from buffer
        log.debug(f'Yielding chunk from {self.rawdata.__class__.__name__} '
                  f'between {self.chunk_time_pre} - {self.chunk_time}')
        maska = records['time'] <= self.chunk_time
        if self.blevel >= 1:
            max_r_time = records['time'].max()
            log.debug(f'Truncating data at sample time {self.chunk_time}, last record time {max_r_time}')
        else:
            log.debug(f'Truncating data at sample time {self.chunk_time}, no record is produced')
        records = records[maska]
        records = strax.sort_by_time(records)  # Do NOT remove this line

        # Yield an appropriate amount of stuff from the truth buffer
        # and mark it as available for writing again

        maskb = (
            self.truth_buffer['fill'] &
            # This condition will always be false if self.truth_buffer['t_first_photon'] == np.nan
            ((self.truth_buffer['t_first_photon'] <= self.chunk_time) |
             # Hence, we need to use this trick to also save these cases (this
             # is what we set the end time to for np.nans)
             (np.isnan(self.truth_buffer['t_first_photon']) &
              (self.truth_buffer['time'] <= self.chunk_time)))
        )
        truth = self.truth_buffer[maskb]   # This is a copy, not a view!

        # Careful here: [maskb]['fill'] = ... does not work
        # numpy creates a copy of the array on the first index.
        # The assignment then goes to the (unused) copy.
        # ['fill'][maskb] leads to a view first, then the advanced
        # assignment works into the original array as expected.
        self.truth_buffer['fill'][maskb] = False

        truth.sort(order='time')
        # Return truth without 'fill' field
        _truth = np.zeros(len(truth), dtype=instruction_dtype + self.truth_dtype)
        for name in _truth.dtype.names:
            _truth[name] = truth[name]
        _truth['time'][~np.isnan(_truth['t_first_photon'])] = \
            _truth['t_first_photon'][~np.isnan(_truth['t_first_photon'])].astype(int)
        _truth.sort(order='time')

        # Oke this will be a bit ugly but it's easy
        if self.config['detector'] == 'XENON1T' or self.config['detector'] == 'XENONnT_neutron_veto':
            yield dict(raw_records=records,
                       truth=_truth)
        elif self.config['detector'] == 'XENONnT':
            yield dict(raw_records_simu=records[records['channel'] < self.config['channel_map']['he'][0]],
                       raw_records_he_simu=records[(records['channel'] >= self.config['channel_map']['he'][0]) &
                                              (records['channel'] <= self.config['channel_map']['he'][-1])],
                       raw_records_aqmon_simu=records[records['channel'] == 800],
                       truth=_truth)

        self.record_buffer[:np.sum(~maska)] = self.record_buffer[:self.blevel][~maska]
        self.blevel = np.sum(~maska)

    def source_finished(self):
        return self.rawdata.source_finished

    @property
    def _n_channels(self):
        return self.config['n_tpc_pmts']


@export
class SRawRecordsFromFaxNT(SimulatorPlugin):
    """
    Plugin which simulates raw_records_simu from fax instructions.
    Only modified provides, and the rest are the same as the one in wfsim.
    """
    provides = ('raw_records_simu', 'raw_records_he_simu', 'raw_records_aqmon_simu', 'truth')
    data_kind = immutabledict(zip(provides, provides))

    def _setup(self):
        self.sim = ChunkRawRecords(self.config)
        self.sim_iter = self.sim(self.instructions)

    def get_instructions(self):
        if self.config['fax_file']:
            assert not self.config['fax_file'].endswith('root'), 'None optical g4 input is deprecated use EPIX instead'
            assert self.config['fax_file'].endswith('csv'), 'Only csv input is supported'
            self.instructions = instruction_from_csv(self.config['fax_file'])
        else:
            self.instructions = rand_instructions(self.config)

    def check_instructions(self):
        # Let below cathode S1 instructions pass but remove S2 instructions
        m = (self.instructions['z'] < - self.config['tpc_length']) & (self.instructions['type'] == 2)
        self.instructions = self.instructions[~m]
        r_instr = np.sqrt(self.instructions['x']**2 + self.instructions['y']**2)

        assert np.all((r_instr<self.config['tpc_radius'])|np.isclose(r_instr,self.config['tpc_radius'])), \
            "Interaction is outside the TPC (radius)"
        assert np.all(self.instructions['z'] < 0.25), \
            "Interaction is outside the TPC (in Z)"
        assert np.all(self.instructions['amp'] > 0), \
            "Interaction has zero size"

    def infer_dtype(self):
        dtype = {data_type: strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)
                 for data_type in self.provides if data_type != 'truth'}

        dtype['truth'] = instruction_dtype + self._truth_dtype
        return dtype

    def compute(self):
        try:
            result = next(self.sim_iter)
        except StopIteration:
            raise RuntimeError("Bug in chunk count computation")
        self._sort_check(result[self.provides[0]])
        # To accomodate nveto raw records, should be the first in provide.

        return {data_type: self.chunk(
            start=self.sim.chunk_time_pre,
            end=self.sim.chunk_time,
            data=result[data_type],
            data_type=data_type) for data_type in self.provides}
