import logging
from typing import Tuple

import numpy as np
import pandas as pd
import strax
import straxen

from fuse.plugin import FuseBasePlugin
from fuse.plugins.pmt_and_daq.pmt_response_and_daq import PMTResponseAndDAQ
from .s_raw_records import NS_NO_INSTRUCTION_BEFORE_CHUNK_END, NS_NO_INSTRUCTION_AFTER_CHUNK_START

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
    physics simulation (in wfsim format) and returns the data in chunks.

    Modified from csv_file_loader: https://github.com/XENONnT/fuse/blob/e538d32a5a0735757a77b1dce31d6f95a379bf4e/fuse/plugins/detector_physics/csv_input.py#L118
    Similar to the case event_rate=0 there: we don't reassign times.
    """

    __version__ = "0.0.0"

    depends_on: Tuple = "raw_records"
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
        help="CSV file (format as in input_type) to read.",
    )
    input_type = straxen.URLConfig(
        track=False,
        infer_type=False,
        help="Input type of CSV file ('fuse' or 'wfsim').",
    )
    ns_no_instruction_after_chunk_start = straxen.URLConfig(
        default=5e7,
        track=False,
        type=(int),
        help="No instruction this amount of time after a chunk starts will be used, "
        "as a safeguard for not getting raw_records_simu out of raw_records chunk time range. ",
    )
    ns_no_instruction_before_chunk_end = straxen.URLConfig(
        default=5e7,
        track=False,
        type=(int),
        help="No instruction this amount of time before a chunk ends will be used, "
        "as a safeguard for not getting raw_records_simu out of raw_records chunk time range. ",
    )

    def setup(self):
        super().setup()
        self.csv_file_reader = SCsvFileLoader(
            input_file=self.input_file,
            input_type=self.input_type,
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
            return self.chunk(start=start, end=end, data=data)#, data_type="geant4_interactions")

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
    """Class to load a CSV file (in wfsim or fuse format) with detector simulation
    instructions."""

    def __init__(
        self,
        input_file,
        input_type,
        random_number_generator,
        ns_no_instruction_before_chunk_end=NS_NO_INSTRUCTION_BEFORE_CHUNK_END,
        ns_no_instruction_after_chunk_start=NS_NO_INSTRUCTION_AFTER_CHUNK_START,
        debug=False,
    ):
        self.input_file = input_file
        self.input_type = input_type
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

        # Translator to translate the wfsim instructions to the fuse format
        if self.input_type == "wfsim":
            self.translator = InstrTranslator(input_format="wfsim", output_format="fuse")
        elif self.input_type == "fuse":
            print(f"No translator defined as input is already {self.input_type} type")
        else:
            raise ValueError(f"Input type {self.input_type} is not defined and should not be possible")

    def output_chunk(self, chunk_start, chunk_end):
        """Load the simulation instructions from the csv file
        and then translate them to the fuse format if neccessary.

        Truncate the instructions to the chunk time range.
        """
        # Load the csv file
        log.warning(f"Loaded detector simulation instructions from a csv file in {self.input_type} format!")
        instructions = self._load_csv_file()
        
        if self.input_type == "wfsim":
            # Translate the wfsim instructions to the fuse format
            log.warning("Translating the wfsim instructions to the fuse format!")
            instructions = self.translator.translate(instructions)
            log.warning("Instructions translated to the fuse format!")

        # truncate instructions to the chunk time range
        log.warning("Truncating instructions to the chunk time range!")
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
        """Load the simulation instructions from a csv file in wfsim format."""
        log.warning(f"Loading detector simulation instructions from {self.input_file} in {self.input_type} format!")
        df = pd.read_csv(self.input_file)

        return df


class InstrTranslator:
    """Class to translate instructions between wfsim and fuse formats."""

    def __init__(self, input_format="wfsim", output_format="fuse"):
        self.input_format = input_format
        self.output_format = output_format

        assert self.input_format in [
            "wfsim",
            "fuse",
        ], "Unknown input format! Choose 'wfsim' or 'fuse'!"
        assert self.output_format in [
            "wfsim",
            "fuse",
        ], "Unknown output format! Choose 'wfsim' or 'fuse'!"

        log.debug("Translating instructions from %s to %s", self.input_format, self.output_format)
        self.translator = self.translator()

    def translator(self):
        if self.input_format == "wfsim" and self.output_format == "fuse":
            return self.translate_wfsim_to_fuse
        elif self.input_format == "fuse" and self.output_format == "wfsim":
            return self.translate_fuse_to_wfsim
        else:
            raise NotImplementedError(
                "Translation from {} to {} is not implemented yet!".format(
                    self.input_format, self.output_format
                )
            )

    def translate(self, instructions):
        return self.translator(instructions)

    def translate_wfsim_to_fuse(self, instructions):
        """Translate the wfsim instructions to the fuse format."""
        # Sort time for sanity
        instructions = instructions.sort_values(by="time")

        # Find cluster and events row by row
        previous_time = 0
        previous_event_number = -1
        cluster_id = 0
        event_id = 0
        first_row = True
        for row in instructions.itertuples():
            new_cluster = False
            new_event = False

            # Check if we have a new cluster or event
            if row.time > previous_time:
                new_cluster = True
            if row.event_number != previous_event_number:
                new_event = True
            if new_event and (not new_cluster):
                raise ValueError("New event without new cluster at time %s!?" % (row.time))
            previous_time = row.time
            previous_event_number = row.event_number

            # Update the previous event number
            if new_event:
                event_id += 1

            # Update the cluster as a new row
            if new_cluster:
                cluster_id += 1
                new_row = {
                    "x": np.float32(row.x),
                    "y": np.float32(row.y),
                    "z": np.float32(row.z),
                    "e_field": np.float32(row.local_field),
                    "ed": np.float32(row.e_dep),
                    "nestid": np.int8(row.recoil),
                    "t": np.int64(row.time),
                    "cluster_id": np.int32(cluster_id),
                    "eventid": np.int32(event_id),
                    "photons": np.int32(0),
                    "electrons": np.int32(0),
                    "excitons": np.int32(0),
                }
                last_row = new_row

            # Assign the number of photons, excitons and electrons to the last row
            if row.type == 1:
                last_row["photons"] = np.int32(row.amp)
                last_row["excitons"] = np.int32(row.n_excitons)
            elif row.type == 2:
                last_row["electrons"] = np.int32(row.amp)
            else:
                raise ValueError("Unknown type %s!" % (row.type))

            # Concatenate the last row to the new instructions
            if first_row:
                rows = [last_row]
                first_row = False
            elif new_cluster:
                rows.append(last_row)

        return pd.DataFrame(rows)

    def translate_fuse_to_wfsim(self, instructions):
        """Translate the fuse instructions to the wfsim format."""
        raise NotImplementedError("Not implemented yet!")
