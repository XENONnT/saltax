import logging
import numpy as np
import pandas as pd
import strax
import straxen
from straxen import units

from fuse.plugin import FuseBasePlugin
from fuse.plugins.detector_physics.csv_input import microphysics_summary_fields, ChunkCsvInput

SALT_TIME_INTERVAL = 2e7  # in unit of ns. The number should be way bigger then full drift time
NS_NO_INSTRUCTION_AFTER_CHUNK_START = 5e7
NS_NO_INSTRUCTION_BEFORE_CHUNK_END = 5e7

export, __all__ = strax.exporter()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
log = logging.getLogger("saltax.plugins.csv_input")


@export
class SChunkCsvInput(FuseBasePlugin):
    """Plugin which reads a CSV file containing instructions for the detector physics simulation and
    returns the data in chunks.

    Modified from csv_file_loader: https://github.com/XENONnT/fuse/blob/e538d32a5a0735757a77b1dce31d6f95a379bf4e/fuse/plugins/detector_physics/csv_input.py#L118  # noqa
    Similar to the case event_rate=0 there: we don't reassign times.

    """

    __version__ = "0.0.1"

    depends_on = "raw_records"
    provides = "microphysics_summary"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        return microphysics_summary_fields + strax.time_fields

    # Config options
    input_file = straxen.URLConfig(
        track=False,
        infer_type=False,
        help="CSV file to read",
    )

    salt_rate = straxen.URLConfig(
        default=units.s / SALT_TIME_INTERVAL,
        infer_type=False,
        help="Rate of salting events",
    )

    ns_no_instruction_after_chunk_start = straxen.URLConfig(
        default=NS_NO_INSTRUCTION_AFTER_CHUNK_START,
        track=False,
        type=int,
        help="No instruction this amount of time after a chunk starts will be used, "
        "as a safeguard for not getting raw_records_simu out of raw_records chunk time range. ",
    )

    ns_no_instruction_before_chunk_end = straxen.URLConfig(
        default=NS_NO_INSTRUCTION_BEFORE_CHUNK_END,
        track=False,
        type=int,
        help="No instruction this amount of time before a chunk ends will be used, "
        "as a safeguard for not getting raw_records_simu out of raw_records chunk time range. ",
    )

    def setup(self):
        super().setup()
        self.csv_file_reader = SCsvFileLoader(
            input_file=self.input_file,
            ns_no_instruction_before_chunk_end=self.ns_no_instruction_before_chunk_end,
            ns_no_instruction_after_chunk_start=self.ns_no_instruction_after_chunk_start,
        )
        self.csv_file_reader.read()

    def compute(self, raw_records, start, end):
        chunk_data = self.csv_file_reader.output_chunk(start, end)
        chunk_data["time"] = chunk_data["t"]
        chunk_data["endtime"] = chunk_data["time"]
        chunk_data = chunk_data.to_records(index=False)
        data = np.zeros(len(chunk_data), dtype=self.dtype)
        strax.copy_to_buffer(chunk_data, data, "_bring_data_into_correct_format")

        return data


class SCsvFileLoader:
    """Class to load a CSV file with detector simulation instructions."""

    def __init__(
        self,
        input_file,
        ns_no_instruction_before_chunk_end=NS_NO_INSTRUCTION_BEFORE_CHUNK_END,
        ns_no_instruction_after_chunk_start=NS_NO_INSTRUCTION_AFTER_CHUNK_START,
    ):
        self.input_file = input_file
        self.ns_no_instruction_before_chunk_end = ns_no_instruction_before_chunk_end
        self.ns_no_instruction_after_chunk_start = ns_no_instruction_after_chunk_start

        # The csv file needs to have these columns:
        _fields = ChunkCsvInput.needed_csv_input_fields()
        self.columns = list(np.dtype(_fields).names)
        self.dtype = _fields + strax.time_fields

    def read(self):
        """Load the simulation instructions from the csv file."""
        self.instructions = self._load_csv_file()

    def output_chunk(self, chunk_start, chunk_end):
        """Load the simulation instructions from the csv file.

        Truncate the instructions to the chunk time range.

        """

        # truncate instructions to the chunk time range
        log.debug("Truncating instructions to the chunk time range!")
        log.debug(
            "We will further truncate the instructions to the range [%d, %d]",
            chunk_start + self.ns_no_instruction_after_chunk_start,
            chunk_end - self.ns_no_instruction_before_chunk_end,
        )

        mask = self.instructions["t"] >= chunk_start + self.ns_no_instruction_after_chunk_start
        mask &= self.instructions["t"] < chunk_end - self.ns_no_instruction_before_chunk_end
        instructions = self.instructions[mask].reset_index(drop=True)

        return instructions

    def _load_csv_file(self):
        """Load the simulation instructions from a csv file."""
        log.debug("Loading detector simulation instructions from a csv file!")
        df = pd.read_csv(self.input_file)

        return df
