import logging
from typing import Tuple

import numpy as np
import pandas as pd
import strax
import straxen

from fuse.plugin import FuseBasePlugin
from fuse.plugins.pmt_and_daq.pmt_response_and_daq import PMTResponseAndDAQ

NS_NO_INSTRUCTION_AFTER_CHUNK_START = 5e7
NS_NO_INSTRUCTION_BEFORE_CHUNK_END = 5e7

export, __all__ = strax.exporter()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.detector_physics.csv_input")


@export
class SPMTResponseAndDAQ(PMTResponseAndDAQ):
    __version__ = "0.0.0"
    provides = "raw_records_simu"
    data_kind = "raw_records_simu"


@export
class SChunkCsvInput(FuseBasePlugin):
    """Plugin which reads a CSV file containing instructions for the detector
    physics simulation and returns the data in chunks.

    Modified from csv_file_loader: https://github.com/XENONnT/fuse/blob/e538d32a5a0735757a77b1dce31d6f95a379bf4e/fuse/plugins/detector_physics/csv_input.py#L118
    Similar to the case event_rate=0 there: we don't reassign times.
    """

    __version__ = "0.0.1"

    depends_on = "raw_records"
    provides = "microphysics_summary"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.TARGET

    # source_done = False

    dtype = [
        (("x position of the cluster [cm]", "x"), np.float32),
        (("y position of the cluster [cm]", "y"), np.float32),
        (("z position of the cluster [cm]", "z"), np.float32),
        (("Number of photons at interaction position", "photons"), np.int32),
        (("Number of electrons at interaction position", "electrons"), np.int32),
        (("Number of excitons at interaction position", "excitons"), np.int32),
        (("Electric field value at the cluster position [V/cm]", "e_field"), np.float32),
        (("Energy of the cluster [keV]", "ed"), np.float32),
        (("NEST interaction type", "nestid"), np.int8),
        (("ID of the cluster", "cluster_id"), np.int32),
    ]
    dtype = dtype + strax.time_fields

    # Config options
    input_file = straxen.URLConfig(
        track=False,
        infer_type=False,
        help="CSV file to read",
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
            random_number_generator=self.rng,
            ns_no_instruction_before_chunk_end=self.ns_no_instruction_before_chunk_end,
            ns_no_instruction_after_chunk_start=self.ns_no_instruction_after_chunk_start,
            debug=self.debug,
        )

    def compute(self, raw_records, start, end):
        try:
            chunk_data, source_done = self.csv_file_reader.output_chunk(start, end)
            chunk_data["time"] = chunk_data["t"]
            chunk_data["endtime"] = chunk_data["time"]
            chunk_data = chunk_data.to_records(index=False)
            data = np.zeros(len(chunk_data), dtype=self.dtype)
            strax.copy_to_buffer(chunk_data, data, "_bring_data_into_correct_format")

            self.source_done = source_done

            # Stick rigorously with raw_records time range
            return self.chunk(start=start, end=end, data=data, data_type="geant4_interactions")

        except StopIteration:
            raise RuntimeError("Bug in chunk building!")

    def source_finished(self):
        return self.source_done

    def is_ready(self, chunk_i):
        """Overwritten to mimic online input plugin.

        Returns False to check source finished; Returns True to get next
        chunk.
        """
        if "ready" not in self.__dict__:
            self.ready = False
        self.ready ^= True  # Flip
        return self.ready


class SCsvFileLoader:
    """Class to load a CSV file with detector simulation
    instructions."""

    def __init__(
        self,
        input_file,
        random_number_generator,
        ns_no_instruction_before_chunk_end=NS_NO_INSTRUCTION_BEFORE_CHUNK_END,
        ns_no_instruction_after_chunk_start=NS_NO_INSTRUCTION_AFTER_CHUNK_START,
        debug=False,
    ):
        self.input_file = input_file
        self.rng = random_number_generator
        self.ns_no_instruction_before_chunk_end = ns_no_instruction_before_chunk_end
        self.ns_no_instruction_after_chunk_start = ns_no_instruction_after_chunk_start
        self.debug = debug

        self.dtype = [
            (("x position of the cluster [cm]", "x"), np.float32),
            (("y position of the cluster [cm]", "y"), np.float32),
            (("z position of the cluster [cm]", "z"), np.float32),
            (("Number of photons at interaction position", "photons"), np.int32),
            (("Number of electrons at interaction position", "electrons"), np.int32),
            (("Number of excitons at interaction position", "excitons"), np.int32),
            (("Electric field value at the cluster position [V/cm]", "e_field"), np.float32),
            (("Energy of the cluster [keV]", "ed"), np.float32),
            (("NEST interaction type", "nestid"), np.int8),
            (("ID of the cluster", "cluster_id"), np.int32),
            (("Time of the interaction", "t"), np.int64),
            (("Geant4 event ID", "eventid"), np.int32),
        ]
        self.dtype = self.dtype + strax.time_fields

        # The csv file needs to have these columns:
        self.columns = [
            "x",
            "y",
            "z",
            "photons",
            "electrons",
            "excitons",
            "e_field",
            "ed",
            "nestid",
            "t",
            "eventid",
            "cluster_id",
        ]

    def output_chunk(self, chunk_start, chunk_end):
        """Load the simulation instructions from the csv file.

        Truncate the instructions to the chunk time range.
        """
        instructions = self._load_csv_file()

        # truncate instructions to the chunk time range
        log.debug("Truncating instructions to the chunk time range!")
        log.debug(
            "We will further truncate the instructions to the range [%d, %d]",
            chunk_start + self.ns_no_instruction_after_chunk_start,
            chunk_end - self.ns_no_instruction_before_chunk_end,
        )

        # See if we have any instructions after the chunk end
        mask_next = instructions["t"] > chunk_end
        if np.any(mask_next):
            source_done = False
        else:
            log.debug("This is the last chunk! No more instructions available!")
            source_done = True

        mask = instructions["t"] >= chunk_start + self.ns_no_instruction_after_chunk_start
        mask &= instructions["t"] < chunk_end - self.ns_no_instruction_before_chunk_end
        instructions = instructions[mask]

        return instructions, source_done

    def _load_csv_file(self):
        """Load the simulation instructions from a csv file."""
        log.debug("Loading detector simulation instructions from a csv file!")
        df = pd.read_csv(self.input_file)

        return df
