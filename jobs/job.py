import saltax
import strax
import straxen
import configparser
from datetime import datetime
import sys
import gc
import os

straxen.print_versions()

now = datetime.now()
_, runid = sys.argv
runid = int(runid)
strrunid = str(runid).zfill(6)

config = configparser.ConfigParser()
config.read("config.ini")
output_folder = str(config.get("job", "output_folder"))
saltax_mode = config.get("job", "saltax_mode")
package = config.get("job", "package")
simu_config_version = config.get("job", "simu_config_version")
generator_name = config.get("job", "generator_name")
recoil = config.getint("job", "recoil")
simu_mode = config.get("job", "simu_mode")
_rate = config.get("job", "rate", fallback=None)
if _rate is not None and _rate.strip():  # Check if rate_value is not just whitespace
    rate = float(_rate)
else:
    rate = None
_en_range = config.get("job", "en_range", fallback=None)
if _en_range is not None and _en_range.strip():  # Check if en_range is not just whitespace
    convert_to_tuple = lambda s: tuple(float(x) if '.' in x else int(x) for x in s.split(','))
    __en_range = convert_to_tuple(_en_range)
else:
    __en_range = None
if (__en_range is not None) and generator_name != "flat":
    en_range = None
    print("You specified en_range = %s, but generator_name = %s. " % (__en_range, generator_name))
    print("en_range will be ignored and be replaced by None. ")
else:
    en_range = __en_range
process_data = config.getboolean("job", "process_data")
process_simu = config.getboolean("job", "process_simu")
skip_records = config.getboolean("job", "skip_records")
storage_to_patch = config.get("job", "storage_to_patch").split(",")
delete_records = config.getboolean("job", "delete_records")

# Determine context for processing
if package == "wfsim":
    context_function = saltax.contexts.sxenonnt
elif package == "fuse":
    context_function = saltax.contexts.fxenonnt
else:
    raise ValueError("Invalid package name %s" % package)

# Determine whether to process events type plugins or just peak types
to_process_dtypes_ev = [
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
to_process_dtypes_se = [
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
if "se" in generator_name:
    to_process_dtypes = to_process_dtypes_se
    print("We use a single electron generator, so we will process:")
else:
    to_process_dtypes = to_process_dtypes_ev
    print("Not a single electron generator, so we will process:")
print(to_process_dtypes)
print("====================")

if not skip_records:
    to_process_dtypes = ["records"] + to_process_dtypes

print("Used time:", datetime.now() - now)
now = datetime.now()

print("====================")
print("Finished importing and config loading, now start to load context.")
print("Now starting %s context for run %d" % (saltax_mode, runid))
if en_range is None:
    st = context_function(
        runid=runid,
        saltax_mode=saltax_mode,
        output_folder=output_folder,
        simu_config_version=simu_config_version,
        generator_name=generator_name,
        recoil=recoil,
        simu_mode=simu_mode,
        rate=rate,
    )
else:
    st = context_function(
        runid=runid,
        saltax_mode=saltax_mode,
        output_folder=output_folder,
        simu_config_version=simu_config_version,
        generator_name=generator_name,
        recoil=recoil,
        simu_mode=simu_mode,
        rate=rate,
        en_range=en_range,
    )
if len(storage_to_patch) and storage_to_patch[0] != "":
    for d in storage_to_patch:
        st.storage.append(strax.DataDirectory(d, readonly=True))

if package == "fuse":
    print("Making microphysics_summary.")
    st.make(strrunid, "microphysics_summary", progress_bar=True)
    print("Done with microphysics_summary.")
    gc.collect()
print("Making raw_records.")
st.make(strrunid, "raw_records_simu", progress_bar=True)
print("Done with raw_records.")
gc.collect()
for dt in to_process_dtypes:
    print("Making %s. " % dt)
    try:
        st.make(strrunid, dt, save=(dt), progress_bar=True)
        print("Done with %s. " % dt)
    except NotImplementedError as e:
        print("The cut_basics for run %d is not implemented. " % runid)
    gc.collect()

print("Used time:", datetime.now() - now)
now = datetime.now()

print(
    "Finished making all the computation for run %d in \
	saltax mode salt. "
    % (runid)
)

if saltax_mode == "salt":
    if process_data:
        print("====================")
        print(
            "Since you specified saltax_mode = salt, \
          	   we will also compute simulation-only and data-only."
        )
        print("Now starting data-only context for run %d" % (runid))
        if en_range is None:
            st = context_function(
                runid=runid,
                saltax_mode="data",
                output_folder=output_folder,
                simu_config_version=simu_config_version,
                generator_name=generator_name,
                recoil=recoil,
                simu_mode=simu_mode,
                rate=rate,
            )
        else:
            st = context_function(
                runid=runid,
                saltax_mode="data",
                output_folder=output_folder,
                simu_config_version=simu_config_version,
                generator_name=generator_name,
                recoil=recoil,
                simu_mode=simu_mode,
                rate=rate,
                en_range=en_range,
            )
        if len(storage_to_patch) and storage_to_patch[0] != "":
            for d in storage_to_patch:
                st.storage.append(strax.DataDirectory(d, readonly=True))

        for dt in to_process_dtypes:
            print("Making %s. " % dt)
            try:
                st.make(strrunid, dt, save=(dt), progress_bar=True)
                print("Done with %s. " % dt)
            except NotImplementedError as e:
                print("The cut_basics for run %d is not implemented. " % runid)
            gc.collect()

        print("Used time:", datetime.now() - now)
        now = datetime.now()

        print(
            "Finished making all the computation for run %d in \
			saltax mode %s. "
            % (runid, "data")
        )
        print("====================")
    else:
        print("You specified process_data = False, so we will not process data.")

    if process_simu:
        print("====================")
        print("Now starting simu-only context for run %d" % (runid))
        if en_range is None:
            st = context_function(
                runid=runid,
                saltax_mode="simu",
                output_folder=output_folder,
                simu_config_version=simu_config_version,
                generator_name=generator_name,
                recoil=recoil,
                simu_mode=simu_mode,
                rate=rate,
            )
        else:
            st = context_function(
                runid=runid,
                saltax_mode="simu",
                output_folder=output_folder,
                simu_config_version=simu_config_version,
                generator_name=generator_name,
                recoil=recoil,
                simu_mode=simu_mode,
                rate=rate,
                en_range=en_range,
            )
        if len(storage_to_patch) and storage_to_patch[0] != "":
            for d in storage_to_patch:
                st.storage.append(strax.DataDirectory(d, readonly=True))

        for dt in to_process_dtypes:
            print("Making %s. " % dt)

            try:
                st.make(strrunid, dt, save=(dt), progress_bar=True)
                print("Done with %s. " % dt)
            except NotImplementedError as e:
                print("The cut_basics for run %d is not implemented. " % runid)
            gc.collect()
        # Manually make pema plugin after
        # st.make(strrunid, "match_acceptance_extended", progress_bar=True)

        print("Used time:", datetime.now() - now)
        now = datetime.now()

        print(
            "Finished making all the computation for run %d in \
			saltax mode %s. "
            % (runid, "simu")
        )
        print("====================")
    else:
        print("You specified process_simu = False, so we will not process simu.")

if delete_records:
    print("Deleting records.")
    records_name = str(st.key_for(strrunid, "records"))
    records_path = os.path.join(output_folder, records_name)
    if os.path.exists(records_path):
        os.rmdir(records_path)
        gc.collect()
        print("Deleted records for run %d in saltax mode salt. " % (runid))
print("====================")


print("Finished all. Exiting.")
