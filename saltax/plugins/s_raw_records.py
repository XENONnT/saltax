import strax
import numpy as np
from wfsim import ChunkRawRecords
from immutabledict import immutabledict
from wfsim.strax_interface import SimulatorPlugin, instruction_dtype, rand_instructions, instruction_from_csv

export, __all__ = strax.exporter()


@export
class RawRecordsFromFaxNT(SimulatorPlugin):
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

