import configparser
from datetime import datetime
import sys
import gc
import os
import saltax
import strax
import straxen


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
    "cuts_basic",
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
    straxen.print_versions()

def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    settings = {
        'output_folder': config.get("job", "output_folder"),
        'saltax_mode': config.get("job", "saltax_mode"),
        'package': config.get("job", "package"),
        'simu_config_version': config.get("job", "simu_config_version"),
        'generator_name': config.get("job", "generator_name"),
        'recoil': config.getint("job", "recoil"),
        'simu_mode': config.get("job", "simu_mode"),
        'rate': float(config.get("job", "rate", fallback=0)) or None,
        'en_range': parse_en_range(config.get("job", "en_range", fallback='')),
        'process_data': config.getboolean("job", "process_data"),
        'process_simu': config.getboolean("job", "process_simu"),
        'skip_records': config.getboolean("job", "skip_records"),
        'storage_to_patch': config.get("job", "storage_to_patch").split(","),
        'delete_records': config.getboolean("job", "delete_records")
    }
    return settings

def parse_en_range(en_range_str):
    if en_range_str.strip():
        return tuple(float(x) if '.' in x else int(x) for x in en_range_str.split(','))
    return None

def create_context(settings, runid):
    context_function = get_context_function(settings['package'])
    st = context_function(
        runid=runid,
        saltax_mode=settings['saltax_mode'],
        output_folder=settings['output_folder'],
        simu_config_version=settings['simu_config_version'],
        generator_name=settings['generator_name'],
        recoil=settings['recoil'],
        simu_mode=settings['simu_mode'],
        rate=settings['rate'] if settings['rate'] else None,
        en_range=settings['en_range'] if settings['en_range'] else None
    )
    for d in settings['storage_to_patch']:
        if d:
            st.storage.append(strax.DataDirectory(d, readonly=True))
    return st

def get_context_function(package):
    if package == "wfsim":
        return saltax.contexts.sxenonnt
    elif package == "fuse":
        return saltax.contexts.fxenonnt
    raise ValueError("Invalid package name %s" % package)

def get_data_types(settings):
    # Decide if it is event level study or not
    if settings['generator_name'] == "se" or settings['generator_name'] == "se_bootstrapped":
        to_process_dtypes = TO_PROCESS_DTYPES_SE
    else:
        to_process_dtypes = TO_PROCESS_DTYPES_EV

    # Decide whether to skip records
    to_process_dtypes = ["raw_records_simu", "records"] + to_process_dtypes if not settings['skip_records'] else to_process_dtypes

    # Decide whether to process microphysics_summary
    to_process_dtypes = to_process_dtypes = ['microphysics_summary'] + to_process_dtypes if settings['package'] == "fuse" else to_process_dtypes

    return to_process_dtypes

def process_data_types(st, strrunid, data_types):
    for dt in data_types:
        print("Making %s. " % dt)
        try:
            st.make(strrunid, dt, save=(dt), progress_bar=True)
            print("Done with %s. " % dt)
        except NotImplementedError as e:
            print("Error for data type %s: %s" % (dt, str(e)))
        gc.collect()

def delete_records_if_needed(settings, runid, st):
    if settings['delete_records']:
        records_name = str(st.key_for(runid, "records"))
        records_path = os.path.join(settings['output_folder'], records_name)
        if os.path.exists(records_path):
            os.rmdir(records_path)
            gc.collect()
            print("Deleted records for run %d in saltax mode salt. " % (runid))

def main():
    print_versions()
    start_time = datetime.now()
    _, runid = sys.argv
    runid = int(runid)
    settings = load_config()

    print("Loading context...")
    st = create_context(settings, runid)
    data_types = get_data_types(settings)
    process_data_types(st, str(runid).zfill(6), data_types)

    # Process data-only mode if required
    if settings['process_data']:
        print("====================")
        print("Now starting data-only context for run %d" % runid)
        settings['saltax_mode'] = 'data'
        st_data = create_context(settings, runid)
        process_data_types(st_data, str(runid).zfill(6), data_types)
        print("Finished processing for data-only mode.")

    # Process simu-only mode if required
    if settings['process_simu']:
        print("====================")
        print("Now starting simu-only context for run %d" % runid)
        settings['saltax_mode'] = 'simu'
        st_simu = create_context(settings, runid)
        process_data_types(st_simu, str(runid).zfill(6), data_types)
        print("Finished processing for simu-only mode.")

    # Delete records if needed
    delete_records_if_needed(settings, runid, st)

    print("====================")
    print("Finished all computations for run %d." % runid)
    print("Total elapsed time:", datetime.now() - start_time)
    print("Exiting.")

if __name__ == "__main__":
    main()
