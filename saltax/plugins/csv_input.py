import os
import logging
import numpy as np
import pandas as pd
import strax
import straxen

from fuse.plugin import FuseBasePlugin
from fuse.plugins.detector_physics.csv_input import microphysics_summary_fields, ChunkCsvInput

import saltax

SALT_TIME_INTERVAL = 2e7  # in unit of ns. The number should be way bigger then full drift time
NO_INSTRUCTION_AFTER_CHUNK_START = int(5e7)
NO_INSTRUCTION_BEFORE_CHUNK_END = int(5e7)

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

    generator_name = straxen.URLConfig(
        default="flat",
        type=str,
        help="Name of generator to use for the CSV input.",
    )

    generator_kwargs = straxen.URLConfig(
        default=dict(),
        track=False,
        type=dict,
        help="Keyword arguments to pass to the generator.",
    )

    no_instruction_after_chunk_start = straxen.URLConfig(
        default=NO_INSTRUCTION_AFTER_CHUNK_START,
        track=False,
        type=int,
        help="No instruction this amount of time after a chunk starts will be used, "
        "as a safeguard for not getting raw_records_simu out of raw_records chunk time range. ",
    )

    no_instruction_before_chunk_end = straxen.URLConfig(
        default=NO_INSTRUCTION_BEFORE_CHUNK_END,
        track=False,
        type=int,
        help="No instruction this amount of time before a chunk ends will be used, "
        "as a safeguard for not getting raw_records_simu out of raw_records chunk time range. ",
    )

    efield_map = straxen.URLConfig(
        cache=True,
        help="Map of the electric field in the detector",
    )

    def infer_dtype(self):
        return microphysics_summary_fields + strax.time_fields

    def setup(self):
        super().setup()
        input_file = self.instruction_generation()
        self.csv_file_reader = SCsvFileLoader(
            input_file=input_file,
            no_instruction_before_chunk_end=self.no_instruction_before_chunk_end,
            no_instruction_after_chunk_start=self.no_instruction_after_chunk_start,
        )
        self.csv_file_reader.read()

    def instruction_generation(self):
        # Specify simulation instructions
        instr_file_name = saltax.instructions.generator.instr_file_name
        input_file = instr_file_name(
            run_id=self.run_id,
            chunk_number=getattr(self, "chunk_number", None),
            **straxen.filter_kwargs(
                instr_file_name,
                {**self.generator_kwargs, "generator_name": self.generator_name},
            ),
        )

        # Try to load instruction from file and generate if not found
        try:
            instr = pd.read_csv(input_file)
            log.info("Loaded instructions from file", input_file)
        except FileNotFoundError:
            log.info(f"Instruction file {input_file} not found. Generating instructions...")
            generator_func = getattr(
                saltax.instructions.generator, "generator_" + self.generator_name
            )
            instr = generator_func(
                run_id=self.run_id,
                **straxen.filter_kwargs(
                    generator_func,
                    {**self.generator_kwargs, "efield_map": self.efield_map},
                ),
            )
            os.makedirs(os.path.dirname(input_file), exist_ok=True)
            pd.DataFrame(instr).to_csv(input_file, index=False)
            log.info(f"Instructions saved to {input_file}")

        return input_file

    def compute(self, raw_records, start, end):
        chunk_data = self.csv_file_reader.output_chunk(start, end)

        # If no data in this chunk, the dtype is not known
        if len(chunk_data) == 0:
            return self.empty_result()

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
        no_instruction_before_chunk_end=NO_INSTRUCTION_BEFORE_CHUNK_END,
        no_instruction_after_chunk_start=NO_INSTRUCTION_AFTER_CHUNK_START,
    ):
        self.input_file = input_file
        self.no_instruction_before_chunk_end = no_instruction_before_chunk_end
        self.no_instruction_after_chunk_start = no_instruction_after_chunk_start

        # The csv file needs to have these columns:
        _fields = ChunkCsvInput.needed_csv_input_fields()
        self.columns = list(np.dtype(_fields).names)
        self.dtype = _fields + strax.time_fields

    def read(self):
        """Load the simulation instructions from the csv file."""
        log.debug("Loading detector simulation instructions from a csv file!")
        self.instructions = pd.read_csv(self.input_file)

    def output_chunk(self, chunk_start, chunk_end):
        """Load the simulation instructions from the csv file.

        Truncate the instructions to the chunk time range.

        """

        # truncate instructions to the chunk time range
        log.debug("Truncating instructions to the chunk time range!")
        log.debug(
            "We will further truncate the instructions to the range [%d, %d]",
            chunk_start + self.no_instruction_after_chunk_start,
            chunk_end - self.no_instruction_before_chunk_end,
        )

        mask = self.instructions["t"] >= chunk_start + self.no_instruction_after_chunk_start
        mask &= self.instructions["t"] < chunk_end - self.no_instruction_before_chunk_end
        instructions = self.instructions[mask].reset_index(drop=True)

        return instructions
