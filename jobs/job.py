import configparser
import time
import sys
import gc
import os
import saltax
import strax
import straxen
from functools import wraps
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


TO_PROCESS_DTYPES_EV = [
    "peaklets",
    "peaklet_classification",
    "merged_s2s",
    "peak_basics",
    "events",
    "peak_positions_mlp",
    "peak_positions_gcn",
    "peak_positions_cnn",
    "event_basics",
    "event_info",
    "event_pattern_fit",
    "event_shadow",
    "event_ambience",
    "event_n_channel",
    "veto_intervals",
#    "cuts_basic",
]
TO_PROCESS_DTYPES_SE = [
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
        "output_folder": config.get("job", "output_folder"),
        "saltax_mode": config.get("job", "saltax_mode"),
        "package": config.get("job", "package"),
        "simu_config_version": config.get("job", "simu_config_version"),
        "generator_name": config.get("job", "generator_name"),
        "recoil": config.getint("job", "recoil"),
        "simu_mode": config.get("job", "simu_mode"),
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


def create_context(settings, runid):
    """Create the context for the given settings and runid, and patch storage
    if needed."""
    context_function = get_context_function(settings["package"])
    st = context_function(
        runid=runid,
        saltax_mode=settings["saltax_mode"],
        output_folder=settings["output_folder"],
        simu_config_version=settings["simu_config_version"],
        generator_name=settings["generator_name"],
        recoil=settings["recoil"],
        simu_mode=settings["simu_mode"],
        rate=settings["rate"] if settings["rate"] else None,
        en_range=settings["en_range"] if settings["en_range"] else None,
        unblind=True
    )
    for d in settings["storage_to_patch"]:
        if d:
            st.storage.append(strax.DataDirectory(d, readonly=True))
    return st


def get_context_function(package):
    """Return the context function for the given package."""
    if package == "fuse":
        return saltax.contexts.sxenonnt
    raise ValueError("Invalid package name %s" % package)


def get_data_types(settings):
    """Return the data types to process based on the generators."""
    # Decide if it is event level study or not
    if settings["generator_name"] == "se" or settings["generator_name"] == "se_bootstrapped":
        to_process_dtypes = TO_PROCESS_DTYPES_SE
    else:
        to_process_dtypes = TO_PROCESS_DTYPES_EV

    # Decide whether to skip records
    to_process_dtypes = (
        ["raw_records_simu", "records"] + to_process_dtypes
        if not settings["skip_records"]
        else to_process_dtypes
    )

    # Decide whether to process microphysics_summary
    to_process_dtypes = (
        ["microphysics_summary"] + to_process_dtypes
        if settings["package"] == "fuse"
        else to_process_dtypes
    )

    return to_process_dtypes


def process_data_types(st, runid, data_types):
    """Process the data types for the given context and runid."""
    for dt in data_types:
        logging.info(f"Making {dt}.")
        try:
            st.make(runid, dt, save=(dt), progress_bar=True)
            logging.info(f"Done with {dt}.")
        except NotImplementedError as e:
            logging.error(f"Error for data type {dt}: {str(e)}")
        gc.collect()


def delete_records_if_needed(settings, runid, st):
    """Delete records if needed."""
    if settings["delete_records"]:
        records_name = str(st.key_for(runid, "records"))
        records_path = os.path.join(settings["output_folder"], records_name)
        if os.path.exists(records_path):
            os.rmdir(records_path)
            gc.collect()
            logging.info(f"Deleted records for run {runid} in saltax mode salt.")


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
    _, runid = sys.argv
    runid = str(runid).zfill(6)

    # Process the saltax desired mode
    logging.info("Loading context...")
    settings = load_config()
    st = create_context(settings, runid)
    data_types = get_data_types(settings)
    print_settings(settings)
    process_data_types(st, runid, data_types)

    # Process data-only mode if required
    if settings["process_data"] and settings["saltax_mode"] == "salt":
        logging.info("====================")
        logging.info(f"Now starting data-only context for run {runid}")
        settings_temp = settings.copy()
        settings_temp["saltax_mode"] = "data"
        st_data = create_context(settings_temp, runid)
        print_settings(settings_temp)
        process_data_types(st_data, runid, data_types)
        logging.info("Finished processing for data-only mode.")

    # Process simu-only mode if required
    if settings["process_simu"] and settings["saltax_mode"] == "salt":
        logging.info("====================")
        logging.info(f"Now starting simu-only context for run {runid}")
        settings_temp = settings.copy()
        settings_temp["saltax_mode"] = "simu"
        st_simu = create_context(settings_temp, runid)
        print_settings(settings_temp)
        process_data_types(st_simu, runid, data_types)
        logging.info("Finished processing for simu-only mode.")

    # Delete records if needed
    delete_records_if_needed(settings, runid, st)

    logging.info("====================")
    logging.info(f"Finished all computations for run {runid}.")
    logging.info("Exiting.")


if __name__ == "__main__":
    main()
