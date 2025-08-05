import os
import sys
import gc
import time
import shutil
import configparser
import logging
from functools import wraps

import strax
import straxen
import saltax

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


TO_PROCESS_DTYPES_EV = [
    "microphysics_summary",
    "peaklets",
    "peaklet_classification",
    "merged_s2s",
    "peak_basics",
    "peak_positions_mlp",
    "peak_positions_gcn",
    "peak_positions_cnn",
    "event_basics",
    "event_info",
    "event_pattern_fit",
    "event_shadow",
    "event_ambience",
    "veto_intervals",
]
TO_PROCESS_DTYPES_SE = [
    "microphysics_summary",
    "peaklets",
    "peaklet_classification",
    "merged_s2s",
    "peak_basics",
    "peak_positions_mlp",
    "peak_positions_cnn",
    "peak_positions_gcn",
    "peak_shadow",
    "peak_ambience",
]


def print_versions():
    """Print the versions of saltax, strax, and straxen."""
    logging.info(straxen.print_versions(["saltax", "strax", "straxen", "fuse", "nestpy"]))


def load_config():
    """Load the configuration file and return the settings."""
    config = configparser.ConfigParser()
    config.read("config.ini")
    settings = {
        "saltax_mode": config.get("job", "saltax_mode"),
        "generator_name": config.get("job", "generator_name"),
        "recoil": config.getint("job", "recoil"),
        "simu_mode": config.get("job", "simu_mode"),
        "output_folder": config.get("job", "output_folder"),
        "corrections_version": config.get("job", "corrections_version"),
        "simulation_config": config.get("job", "simulation_config"),
        "rate": float(config.get("job", "rate", fallback=0)) or None,
        "en_range": parse_en_range(config.get("job", "en_range", fallback="")),
        "process_data": config.getboolean("job", "process_data"),
        "process_simu": config.getboolean("job", "process_simu"),
        "skip_records": config.getboolean("job", "skip_records"),
        "storage_to_patch": config.get("job", "storage_to_patch").split(","),
        "delete_records": config.getboolean("job", "delete_records"),
    }
    return settings


def print_settings(settings):
    """Print the settings in a table format."""
    logging.info("====================================")
    logging.info("Settings:")
    for k, v in settings.items():
        logging.info(f"{k}: {v}")
    logging.info("====================================")


def parse_en_range(en_range_str):
    """Parse the energy range string and return a tuple of floats."""
    if en_range_str.strip():
        return tuple(float(x) if "." in x else int(x) for x in en_range_str.split(","))
    return None


def create_context(settings, run_id):
    """Create the context for the given settings and run_id, and patch storage if needed."""
    st = saltax.contexts.sxenonnt(
        run_id=run_id,
        saltax_mode=settings["saltax_mode"],
        generator_name=settings["generator_name"],
        recoil=settings["recoil"],
        simu_mode=settings["simu_mode"],
        output_folder=settings["output_folder"],
        corrections_version=settings["corrections_version"],
        simulation_config=settings["simulation_config"],
        rate=settings["rate"] if settings["rate"] else None,
        en_range=settings["en_range"] if settings["en_range"] else None,
        unblind=True,
    )
    for d in settings["storage_to_patch"]:
        if d:
            st.storage.append(strax.DataDirectory(d, readonly=True))
    return st


def get_data_types(settings):
    """Return the data types to process based on the generators."""
    # Decide if it is event level study or not
    if settings["generator_name"] == "se" or settings["generator_name"] == "se_bootstrapped":
        to_process_dtypes = TO_PROCESS_DTYPES_SE
    else:
        to_process_dtypes = TO_PROCESS_DTYPES_EV

    # Decide whether to skip records
    if settings["skip_records"]:
        return to_process_dtypes
    else:
        return ["raw_records_simu", "records"] + to_process_dtypes


def process_data_types(st, run_id, data_types):
    """Process the data types for the given context and run_id."""
    for dt in data_types:
        logging.info(f"Making {dt}.")
        try:
            st.make(run_id, dt, save=dt, progress_bar=True)
            logging.info(f"Done with {dt}.")
        except NotImplementedError as e:
            logging.error(f"Error for data type {dt}: {str(e)}")
        gc.collect()


def delete_records_if_needed(settings, run_id, st):
    """Delete records if needed."""
    if settings["delete_records"]:
        records_name = str(st.key_for(run_id, "records"))
        records_path = os.path.join(settings["output_folder"], records_name)
        if os.path.exists(records_path):
            shutil.rmtree(records_path)
            gc.collect()
            logging.info(f"Deleted records for run {run_id} in saltax mode salt.")


def timeit(func):
    """Decorator to measure the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"Total elapsed time for {func.__name__}: {elapsed_time:.2f} seconds.")
        return result

    return wrapper


@timeit
def main():
    print_versions()
    _, run_id = sys.argv
    run_id = str(run_id).zfill(6)

    # Process the saltax desired mode
    logging.info("Loading context...")
    settings = load_config()
    st = create_context(settings, run_id)
    data_types = get_data_types(settings)
    print_settings(settings)
    process_data_types(st, run_id, data_types)

    # Process data-only mode if required
    if settings["process_data"] and settings["saltax_mode"] == "salt":
        logging.info("====================")
        logging.info(f"Now starting data-only context for run {run_id}")
        settings_temp = settings.copy()
        settings_temp["saltax_mode"] = "data"
        st_data = create_context(settings_temp, run_id)
        print_settings(settings_temp)
        process_data_types(st_data, run_id, data_types)
        logging.info("Finished processing for data-only mode.")

    # Process simu-only mode if required
    if settings["process_simu"] and settings["saltax_mode"] == "salt":
        logging.info("====================")
        logging.info(f"Now starting simu-only context for run {run_id}")
        settings_temp = settings.copy()
        settings_temp["saltax_mode"] = "simu"
        st_simu = create_context(settings_temp, run_id)
        print_settings(settings_temp)
        process_data_types(st_simu, run_id, data_types)
        logging.info("Finished processing for simu-only mode.")

    # Delete records if needed
    delete_records_if_needed(settings, run_id, st)

    logging.info("====================")
    logging.info(f"Finished all computations for run {run_id}.")
    logging.info("Exiting.")


if __name__ == "__main__":
    main()
