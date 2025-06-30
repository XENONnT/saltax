import os
from glob import glob
from tabulate import tabulate
from itertools import cycle
import numpy as np
from scipy.stats import binomtest
import matplotlib.pyplot as plt

import utilix
import strax
import saltax

ALL_CUTS_MINIMAL = [
    "cut_daq_veto",
    "cut_interaction_exists",
    "cut_main_is_valid_triggering_peak",
    "cut_run_boundaries",
]
ALL_CUTS = ALL_CUTS_MINIMAL + [
    "cut_s1_area_fraction_top",
    "cut_s1_max_pmt",
    "cut_s1_pattern_bottom",
    "cut_s1_pattern_top",
    "cut_s1_single_scatter",
    "cut_s1_tightcoin_3fold",
    "cut_s1_width",
    "cut_s2_pattern",
    "cut_s2_recon_pos_diff",
    "cut_s2_single_scatter",
    "cut_s2_width",
    "cut_cs2_area_fraction_top",
    "cut_shadow",
    "cut_ambience",
]
ALL_CUTS_EXCEPT_S2PatternS1Width = ALL_CUTS_MINIMAL + [
    "cut_s1_area_fraction_top",
    "cut_s1_max_pmt",
    "cut_s1_pattern_bottom",
    "cut_s1_pattern_top",
    "cut_s1_single_scatter",
    "cut_s1_tightcoin_3fold",
    "cut_s2_recon_pos_diff",
    "cut_s2_single_scatter",
    "cut_s2_width",
    "cut_cs2_area_fraction_top",
    "cut_shadow",
    "cut_ambience",
]
AmBe_CUTS_EXCEPT_S2Pattern = ALL_CUTS_MINIMAL + [
    "cut_s1_area_fraction_top",
    "cut_s1_max_pmt",
    "cut_s1_pattern_bottom",
    "cut_s1_pattern_top",
    "cut_s1_single_scatter",
    "cut_s1_tightcoin_3fold",
    "cut_s1_width",
    "cut_s2_recon_pos_diff",
    "cut_s2_single_scatter",
    "cut_s2_width",
    "cut_cs2_area_fraction_top",
]
AmBe_CUTS_EXCEPT_S2PatternS1Width = ALL_CUTS_MINIMAL + [
    "cut_s1_area_fraction_top",
    "cut_s1_max_pmt",
    "cut_s1_pattern_bottom",
    "cut_s1_pattern_top",
    "cut_s1_single_scatter",
    "cut_s1_tightcoin_3fold",
    "cut_s2_recon_pos_diff",
    "cut_s2_single_scatter",
    "cut_s2_width",
    "cut_cs2_area_fraction_top",
]
AmBe_CUTS = ALL_CUTS_MINIMAL + [
    "cut_s1_area_fraction_top",
    "cut_s1_max_pmt",
    "cut_s1_pattern_bottom",
    "cut_s1_pattern_top",
    "cut_s1_single_scatter",
    "cut_s1_tightcoin_3fold",
    "cut_s1_width",
    "cut_s2_pattern",
    "cut_s2_recon_pos_diff",
    "cut_s2_single_scatter",
    "cut_s2_width",
    "cut_cs2_area_fraction_top",
]


def find_runs_with_rawdata(
    rawdata_folders=[
        "/project/lgrandi/yuanlq/salt/raw_records",
        "/scratch/midway2/yuanlq/salt/raw_records",
        "/scratch/midway3/yuanlq/salt/raw_records",
    ]
):
    # Find the files that correspond to strax data
    files_found = []
    for folder in rawdata_folders:
        _files_found = glob(os.path.join(folder, "0*"))
        files_found += _files_found

    # Find the runs that have standard raw_records available
    runs = []
    for f in files_found:
        _f = os.path.basename(f)
        runid, datatype, shash = _f.split("-")
        if datatype == "raw_records" and shash == "rfzvpzj4mf":
            runs.append(runid)
    runs = np.array(runs)
    return runs


def is_stored_dtypes(st, runid, dtypes):
    """Check if all dtypes are stored for a run.

    :param st: saltax context
    :param runid: runid
    :param dtypes: list of dtypes
    :return: True if all dtypes are stored, False otherwise

    """
    if not len(dtypes):
        return True
    else:
        for dtype in dtypes:
            if not st.is_stored(runid, dtype):
                return False
        return True


def sort_runs(runs):
    """Sort the runs in time order based on runid :param runs: list of runs' str.

    :return: ordered runlist based on runid number

    """
    runs_number = []
    for r in runs:
        runs_number.append(int(r))
    return np.array(runs)[np.argsort(runs_number)]


def get_available_runs(
    runs,
    st_salt,
    st_simu,
    salt_available=["peak_basics", "peak_positions_mlp"],
    simu_available=["peak_basics", "peak_positions_mlp"],
):
    """Print out available runs for both salt and simu modes.

    :param runs: list of runs.
    :param st_salt: saltax context for salt mode
    :param st_simu: saltax context for simu mode
    :param salt_available: list of available dtypes for salt mode
    :param simu_available: list of available dtypes for simu mode

    """
    rundb = utilix.rundb.xent_collection()
    # Find run modes and duration correspondingly
    modes = []
    durations = []
    for run in runs:
        query = {"number": int(run)}
        doc = rundb.find_one(query)

        # get mode
        mode = doc["mode"]
        # duration
        td = doc["end"] - doc["start"]
        td_min = int(td.total_seconds() / 60)

        modes.append(mode)
        durations.append(td_min)
    modes = np.array(modes)
    durations = np.array(durations)

    # build dictionaries for modes and runs
    modes_dict = {}
    for mode in np.unique(modes):
        modes_dict[mode] = runs[modes == mode]
    durations_dict = {}
    for i, run in enumerate(runs):
        durations_dict[run] = durations[i]

    # Prepare data for tabulate
    available_runs = []
    table_data = []
    for mode, runids in modes_dict.items():
        for runid in runids:
            if is_stored_dtypes(st_salt, runid, salt_available) and is_stored_dtypes(
                st_simu, runid, simu_available
            ):
                duration = durations_dict.get(runid, "N/A")  # Get duration or 'N/A' if not found
                table_data.append([mode, runid, duration])
                available_runs.append(runid)

    # Print table using tabulate
    print(tabulate(table_data, headers=["mode", "runid", "duration [min]"]))
    print("=============================")
    print("The runs below are available:")
    print(available_runs)
    print("=============================")


def load_peaks(runs, st_salt, st_simu, plugins=("peak_basics", "peak_positions_mlp"), **kwargs):
    """Load peaks from the runs and find matching indices for salted and simulated peaks.

    :param runs: list of runs.
    :param st_salt: saltax context for salt mode
    :param st_simu: saltax context for simu mode
    :param plugins: plugins to be loaded, default to ('peak_basics', 'peak_positions_mlp')
    :param kwargs: arguments for saltax.match_peaks, i.e. window_length
    :return: peaks_simu: peaks from simulated dataset
    :return: peaks_salt: peaks from sprinkled dataset
    :return: inds_dict: dictionary of indices of peaks from sprinkled or filtered simulated dataset,
        regarding matching peaks

    """
    # Order runs so we have monotonically increasing time stamps
    runs = sort_runs(runs)

    # Initialize the dictionary to store the indices
    inds_dict = {
        "ind_salt_peak_found": np.array([], dtype=np.int32),
        "ind_simu_peak_found": np.array([], dtype=np.int32),
        "ind_simu_peak_lost": np.array([], dtype=np.int32),
        "ind_salt_peak_split": np.array([], dtype=np.int32),
        "ind_simu_peak_split": np.array([], dtype=np.int32),
    }

    len_simu_so_far = 0
    len_salt_so_far = 0
    for i, run in enumerate(runs):
        print(f"Loading run {run}")

        # Load plugins for both salt and simu
        peaks_simu_i = st_simu.get_array(run, plugins, progress_bar=False)
        peaks_salt_i = st_salt.get_array(run, plugins, progress_bar=False)

        # Get matching result
        (
            ind_salt_peak_found_i,
            ind_simu_peak_found_i,
            ind_simu_peak_lost_i,
            ind_salt_peak_split_i,
            ind_simu_peak_split_i,
        ) = saltax.match_peaks(peaks_simu_i, peaks_salt_i, **kwargs)

        # Load the indices into the dictionary
        inds_dict["ind_salt_peak_found"] = np.concatenate(
            (inds_dict["ind_salt_peak_found"], ind_salt_peak_found_i + len_salt_so_far)
        )
        inds_dict["ind_simu_peak_found"] = np.concatenate(
            (inds_dict["ind_simu_peak_found"], ind_simu_peak_found_i + len_simu_so_far)
        )
        inds_dict["ind_simu_peak_lost"] = np.concatenate(
            (inds_dict["ind_simu_peak_lost"], ind_simu_peak_lost_i + len_simu_so_far)
        )
        inds_dict["ind_salt_peak_split"] = np.concatenate(
            (inds_dict["ind_salt_peak_split"], ind_salt_peak_split_i + len_salt_so_far)
        )
        inds_dict["ind_simu_peak_split"] = np.concatenate(
            (inds_dict["ind_simu_peak_split"], ind_simu_peak_split_i + len_simu_so_far)
        )

        # Concatenate the peaks
        if i == 0:
            peaks_simu = peaks_simu_i
            peaks_salt = peaks_salt_i
        else:
            peaks_simu = np.concatenate((peaks_simu, peaks_simu_i))
            peaks_salt = np.concatenate((peaks_salt, peaks_salt_i))

        # Update the length of the peaks
        len_simu_so_far += len(peaks_simu_i)
        len_salt_so_far += len(peaks_salt_i)

    return peaks_simu, peaks_salt, inds_dict


def load_events(runs, st_salt, st_simu, plugins=("event_info", "cuts_basic"), **kwargs):
    """Load events from the runs and do basic filtering suggeted by saltax.match_events :param runs:
    list of runs.

    :param st_salt: saltax context for salt mode
    :param st_simu: saltax context for simu mode
    :param plugins: plugins to be loaded, default to ('event_info', 'cuts_basic')
    :param kwargs: arguments for saltax.match_events, i.e. event_window_fuzz,
    :return: events_simu: events from simulated dataset, filtered out those who miss S1
    :return: events_salt: events from sprinkled dataset
    :return inds_dict: dictionary of indices of events from sprinkled or filtered simulated dataset,
        regarding matching events or s1 or s2

    """
    # Order runs so we have monotonically increasing time stamps
    runs = sort_runs(runs)

    # Initialize the dictionary to store the indices
    inds_dict = {
        "ind_salt_event_found": np.array([], dtype=np.int32),
        "ind_salt_s1_found": np.array([], dtype=np.int32),
        "ind_salt_s1_made_alt": np.array([], dtype=np.int32),
        "ind_salt_s2_found": np.array([], dtype=np.int32),
        "ind_salt_s2_made_alt": np.array([], dtype=np.int32),
        "ind_simu_event_found": np.array([], dtype=np.int32),
        "ind_simu_s1_found": np.array([], dtype=np.int32),
        "ind_simu_s1_made_alt": np.array([], dtype=np.int32),
        "ind_simu_s2_found": np.array([], dtype=np.int32),
        "ind_simu_s2_made_alt": np.array([], dtype=np.int32),
        "ind_simu_event_lost": np.array([], dtype=np.int32),
        "ind_simu_event_split": np.array([], dtype=np.int32),
    }

    len_simu_so_far = 0
    len_salt_so_far = 0
    for i, run in enumerate(runs):
        print(f"Loading run {run}")

        # Load plugins for both salt and simu
        events_simu_i = st_simu.get_array(run, plugins, progress_bar=False)
        events_salt_i = st_salt.get_array(run, plugins, progress_bar=False)

        # Get matching result
        (
            events_simu_filtered_i,
            ind_salt_event_found_i,
            ind_simu_event_found_i,
            ind_simu_event_lost_i,
            ind_simu_event_split_i,
            ind_salt_s1_found_i,
            ind_simu_s1_found_i,
            ind_salt_s1_made_alt_i,
            ind_simu_s1_made_alt_i,
            ind_salt_s2_found_i,
            ind_simu_s2_found_i,
            ind_salt_s2_made_alt_i,
            ind_simu_s2_made_alt_i,
        ) = saltax.match_events(events_simu_i, events_salt_i, **kwargs)

        # Load the indices into the dictionary
        inds_dict["ind_salt_event_found"] = np.concatenate(
            (inds_dict["ind_salt_event_found"], ind_salt_event_found_i + len_salt_so_far)
        )
        inds_dict["ind_salt_s1_found"] = np.concatenate(
            (inds_dict["ind_salt_s1_found"], ind_salt_s1_found_i + len_salt_so_far)
        )
        inds_dict["ind_salt_s1_made_alt"] = np.concatenate(
            (inds_dict["ind_salt_s1_made_alt"], ind_salt_s1_made_alt_i + len_salt_so_far)
        )
        inds_dict["ind_salt_s2_found"] = np.concatenate(
            (inds_dict["ind_salt_s2_found"], ind_salt_s2_found_i + len_salt_so_far)
        )
        inds_dict["ind_salt_s2_made_alt"] = np.concatenate(
            (inds_dict["ind_salt_s2_made_alt"], ind_salt_s2_made_alt_i + len_salt_so_far)
        )
        inds_dict["ind_simu_event_found"] = np.concatenate(
            (inds_dict["ind_simu_event_found"], ind_simu_event_found_i + len_simu_so_far)
        )
        inds_dict["ind_simu_s1_found"] = np.concatenate(
            (inds_dict["ind_simu_s1_found"], ind_simu_s1_found_i + len_simu_so_far)
        )
        inds_dict["ind_simu_s1_made_alt"] = np.concatenate(
            (inds_dict["ind_simu_s1_made_alt"], ind_simu_s1_made_alt_i + len_simu_so_far)
        )
        inds_dict["ind_simu_s2_found"] = np.concatenate(
            (inds_dict["ind_simu_s2_found"], ind_simu_s2_found_i + len_simu_so_far)
        )
        inds_dict["ind_simu_s2_made_alt"] = np.concatenate(
            (inds_dict["ind_simu_s2_made_alt"], ind_simu_s2_made_alt_i + len_simu_so_far)
        )
        inds_dict["ind_simu_event_lost"] = np.concatenate(
            (inds_dict["ind_simu_event_lost"], ind_simu_event_lost_i + len_simu_so_far)
        )
        inds_dict["ind_simu_event_split"] = np.concatenate(
            (inds_dict["ind_simu_event_split"], ind_simu_event_split_i + len_simu_so_far)
        )

        # Concatenate the events
        if i == 0:
            events_simu = events_simu_filtered_i
            events_salt = events_salt_i
        else:
            events_simu = np.concatenate((events_simu, events_simu_filtered_i))
            events_salt = np.concatenate((events_salt, events_salt_i))

        # Update the length of the events
        len_simu_so_far += len(events_simu_filtered_i)
        len_salt_so_far += len(events_salt_i)

    return events_simu, events_salt, inds_dict


def compare_templates(
    events_salt_matched_to_simu,
    events_simu_matched_to_salt,
    n_bins=31,
    title="Ambience Interference in SR0 AmBe",
):
    """Visually compare the cs1-cs2 templates of salted and simulated events.

    :param events_salt_matched_to_simu: events from saltax matched to simulation, with equal length
    :param events_simu_matched_to_salt: events from simulation matched to saltax, with equal length
    :param n_bins: number of bins for cs1
    :param title: title of the plot

    """
    cs1_bins = np.linspace(0, 100, n_bins)
    salt_med = []
    simu_med = []
    salt_1sig_u = []
    simu_1sig_u = []
    salt_1sig_l = []
    simu_1sig_l = []
    salt_2sig_u = []
    simu_2sig_u = []
    salt_2sig_l = []
    simu_2sig_l = []

    for i in range(n_bins - 1):
        selected_salt = events_salt_matched_to_simu[
            (events_salt_matched_to_simu["cs1"] >= cs1_bins[i])
            * (events_salt_matched_to_simu["cs1"] < cs1_bins[i + 1])
        ]
        salt_med.append(np.median(selected_salt["cs2"][selected_salt["cs2"] > 0]))
        salt_1sig_l.append(np.percentile(selected_salt["cs2"][selected_salt["cs2"] > 0], 16.5))
        salt_1sig_u.append(np.percentile(selected_salt["cs2"][selected_salt["cs2"] > 0], 83.5))
        salt_2sig_l.append(np.percentile(selected_salt["cs2"][selected_salt["cs2"] > 0], 2.5))
        salt_2sig_u.append(np.percentile(selected_salt["cs2"][selected_salt["cs2"] > 0], 97.5))

        selected_simu = events_simu_matched_to_salt[
            (events_simu_matched_to_salt["cs1"] >= cs1_bins[i])
            * (events_simu_matched_to_salt["cs1"] < cs1_bins[i + 1])
        ]
        simu_med.append(np.median(selected_simu["cs2"][selected_simu["cs2"] > 0]))
        simu_1sig_l.append(np.percentile(selected_simu["cs2"][selected_simu["cs2"] > 0], 16.5))
        simu_1sig_u.append(np.percentile(selected_simu["cs2"][selected_simu["cs2"] > 0], 83.5))
        simu_2sig_l.append(np.percentile(selected_simu["cs2"][selected_simu["cs2"] > 0], 2.5))
        simu_2sig_u.append(np.percentile(selected_simu["cs2"][selected_simu["cs2"] > 0], 97.5))

    cs1_coord = np.linspace(0, 100, n_bins - 1) + 100 / 2 / (n_bins - 1)

    plt.figure(dpi=150)
    plt.scatter(
        events_salt_matched_to_simu["cs1"],
        events_salt_matched_to_simu["cs2"],
        s=0.5,
        label="Sprinkled",
        alpha=0.2,
    )
    plt.scatter(
        events_simu_matched_to_salt["cs1"],
        events_simu_matched_to_salt["cs2"],
        s=0.5,
        label="Simulated",
        alpha=0.2,
    )
    plt.plot(cs1_coord, salt_med, color="tab:blue", label="Sprk Median")
    plt.plot(cs1_coord, simu_med, color="tab:red", label="Simu Median")
    plt.plot(cs1_coord, salt_1sig_l, color="tab:blue", label="Sprk 1sig", linestyle="dashed")
    plt.plot(cs1_coord, salt_1sig_u, color="tab:blue", linestyle="dashed")
    plt.plot(
        cs1_coord, salt_2sig_l, color="tab:blue", label="Sprk 2sig", linestyle="dashed", alpha=0.5
    )
    plt.plot(cs1_coord, salt_2sig_u, color="tab:blue", linestyle="dashed", alpha=0.5)
    plt.plot(cs1_coord, simu_1sig_l, color="tab:red", label="Simu 1sig", linestyle="dashed")
    plt.plot(cs1_coord, simu_1sig_u, color="tab:red", linestyle="dashed")
    plt.plot(
        cs1_coord, simu_2sig_l, color="tab:red", label="Simu 2sig", linestyle="dashed", alpha=0.5
    )
    plt.plot(cs1_coord, simu_2sig_u, color="tab:red", linestyle="dashed", alpha=0.5)

    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 6500)
    plt.xlabel("CS1 [PE]")
    plt.ylabel("CS2 [PE]")
    plt.title(title)
    plt.show()


def apply_n_minus_1_cuts(events_with_cuts, cut_oi, all_cuts=ALL_CUTS_EXCEPT_S2PatternS1Width):
    """Apply N-1 cuts to the events, where N is the number of cuts.

    :param events_with_cuts: events with cuts
    :param cut_oi: the cut to be left out for examination
    :param all_cuts: all cuts

    """
    other_cuts = [cut for cut in all_cuts if cut != cut_oi]
    mask = np.ones(len(events_with_cuts), dtype=bool)

    for cut in other_cuts:
        mask &= events_with_cuts[cut]

    return mask


def apply_single_cut(events_with_cuts, cut_oi, all_cuts=None):
    """Apply a single cut to the events.

    :param events_with_cuts: events with cuts
    :param cut_oi: the cut to be applied
    :param all_cuts: pseudo parameter, not really used

    """
    mask = np.ones(len(events_with_cuts), dtype=bool)
    for cut in [cut_oi]:
        mask &= events_with_cuts[cut]
    return mask


def apply_cut_lists(events_with_cuts, all_cuts=ALL_CUTS_EXCEPT_S2PatternS1Width):
    """Apply a list of cuts to the events.

    :param events_with_cuts: events with cuts
    :param all_cuts: list of cuts to be applied

    """
    mask = np.ones(len(events_with_cuts), dtype=bool)
    for cut in all_cuts:
        mask &= events_with_cuts[cut]
    return mask


def get_n_minus_1_cut_acc(
    events_salt_matched_to_simu, events_simu_matched_to_salt, all_cut_list=ALL_CUTS
):
    """Get a text table of acceptance of N-1 cut acceptance for each cut.

    :param events_salt_matched_to_simu: events from saltax matched to simulation, with equal length
    :param events_simu_matched_to_salt: events from simulation matched to saltax, with equal length
    :param all_cut_list: list of all cuts

    """
    mask_salt_all_cuts = apply_cut_lists(events_salt_matched_to_simu, all_cuts=all_cut_list)
    mask_simu_all_cuts = apply_cut_lists(events_simu_matched_to_salt, all_cuts=all_cut_list)

    # Initialize a list to store your rows
    table_data = []

    # Loop over each cut and calculate the acceptance values
    for cut_oi in all_cut_list:
        mask_salt_except_cut_oi = apply_n_minus_1_cuts(
            events_salt_matched_to_simu, cut_oi, all_cuts=all_cut_list
        )
        mask_simu_except_cut_oi = apply_n_minus_1_cuts(
            events_simu_matched_to_salt, cut_oi, all_cuts=all_cut_list
        )
        acceptance_salt = (
            int(np.sum(mask_salt_all_cuts) / np.sum(mask_salt_except_cut_oi) * 100) / 100
        )
        acceptance_simu = (
            int(np.sum(mask_simu_all_cuts) / np.sum(mask_simu_except_cut_oi) * 100) / 100
        )

        # Add a row for each cut
        table_data.append([cut_oi, acceptance_salt, acceptance_simu])

    # Define the headers
    headers = ["Cut Name", "Acceptance in Sprk", "Acceptance in Simu"]

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def get_single_cut_acc(
    events_salt_matched_to_simu, events_simu_matched_to_salt, all_cut_list=ALL_CUTS
):
    """Get a text table of acceptance of single cut acceptance for each cut.

    :param events_salt_matched_to_simu: events from saltax matched to simulation, with equal length
    :param events_simu_matched_to_salt: events from simulation matched to saltax, with equal length
    :param all_cut_list: list of all cuts

    """
    mask_salt_no_cuts = np.ones(len(events_salt_matched_to_simu), dtype=bool)
    mask_simu_no_cuts = np.ones(len(events_simu_matched_to_salt), dtype=bool)

    # Initialize a list to store your rows
    table_data = []

    # Loop over each cut and calculate the acceptance values
    for cut_oi in all_cut_list:
        mask_salt_except_cut_oi = events_salt_matched_to_simu[cut_oi]
        mask_simu_except_cut_oi = events_simu_matched_to_salt[cut_oi]
        acceptance_salt = (
            int(np.sum(mask_salt_except_cut_oi) / np.sum(mask_salt_no_cuts) * 100) / 100
        )
        acceptance_simu = (
            int(np.sum(mask_simu_except_cut_oi) / np.sum(mask_simu_no_cuts) * 100) / 100
        )

        # Add a row for each cut
        table_data.append([cut_oi, acceptance_salt, acceptance_simu])

    # Define the headers
    headers = ["Cut Name", "Acceptance in Sprk", "Acceptance in Simu"]

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def get_cut_eff(
    events,
    all_cut_list=ALL_CUTS,
    n_bins=31,
    coord="cs1",
    plot=True,
    indv_cut_type="n_minus_1",
    title="N-1 Cut Acceptance Measured in SR1 AmBe",
    bbox_to_anchor=(0.5, 1.50),
    bin_range=None,
):
    """Get the acceptance with corresponding Clopper-Pearson uncertainty of each cut, as a function
    of a coordinate.

    :param events: events
    :param all_cut_list: list of all cuts
    :param n_bins: number of bins for the coordinate
    :param coord: coordinate to be binned, default to 'cs1'
    :param plot: whether to plot the acceptance, default to True
    :param indv_cut_type: type of cut to be applied, default to
        'n_minus_1', can also be 'single'
    :param title: title of the plot
    :param bbox_to_anchor: position of the legend, default to (0.5,
        1.50)
    :param bin_range: range of the coordinate, default to None
    :return: a dictionary of acceptance values

    """
    coord_units = {"s1_area": "[PE]", "s2_area": "[PE]", "cs1": "[PE]", "cs2": "[PE]", "z": "[cm]"}
    if bin_range is not None:
        bins = np.linspace(bin_range[0], bin_range[1], n_bins)
    else:
        if coord == "cs1" or coord == "s1_area":
            bins = np.linspace(0, 100, n_bins)
        elif coord == "cs2" or coord == "s2_area":
            bins = np.linspace(200, 3000, n_bins)
        elif coord == "z":
            bins = np.linspace(-134, -13, n_bins)
        else:
            raise NotImplementedError

    result_dict = {}
    for cut in all_cut_list:
        result_dict[cut] = np.zeros(n_bins - 1)
        result_dict[cut + "_upper"] = np.zeros(n_bins - 1)
        result_dict[cut + "_lower"] = np.zeros(n_bins - 1)
    result_dict["all_cuts"] = np.zeros(n_bins - 1)
    result_dict["all_cuts_upper"] = np.zeros(n_bins - 1)
    result_dict["all_cuts_lower"] = np.zeros(n_bins - 1)
    result_dict[coord] = np.zeros(n_bins - 1)

    for i in range(n_bins - 1):
        mid_coord = (bins[i] + bins[i + 1]) / 2
        result_dict[coord][i] = mid_coord

        selected_events = events[(events[coord] >= bins[i]) * (events[coord] < bins[i + 1])]
        selected_events_all_cut = selected_events[apply_cut_lists(selected_events, all_cut_list)]
        interval = binomtest(len(selected_events_all_cut), len(selected_events)).proportion_ci()
        result_dict["all_cuts"][i] = len(selected_events_all_cut) / len(selected_events)
        result_dict["all_cuts_upper"][i] = interval.high
        result_dict["all_cuts_lower"][i] = interval.low

        for cut_oi in all_cut_list:
            if indv_cut_type == "n_minus_1":
                selected_events_cut_oi = selected_events[
                    apply_n_minus_1_cuts(selected_events, cut_oi, all_cut_list)
                ]
                # Efficiency curves with Clopper-Pearson uncertainty estimation
                interval = binomtest(
                    len(selected_events_all_cut), len(selected_events_cut_oi)
                ).proportion_ci()
                result_dict[cut_oi][i] = len(selected_events_all_cut) / len(selected_events_cut_oi)
                result_dict[cut_oi + "_upper"][i] = interval.high
                result_dict[cut_oi + "_lower"][i] = interval.low
            elif indv_cut_type == "single":
                selected_events_cut_oi = selected_events[apply_single_cut(selected_events, cut_oi)]
                # Efficiency curves with Clopper-Pearson uncertainty estimation
                interval = binomtest(
                    len(selected_events_cut_oi), len(selected_events)
                ).proportion_ci()
                result_dict[cut_oi][i] = len(selected_events_cut_oi) / len(selected_events)
                result_dict[cut_oi + "_upper"][i] = interval.high
                result_dict[cut_oi + "_lower"][i] = interval.low
            else:
                raise NotImplementedError

    if plot:
        colors = plt.cm.rainbow(
            np.linspace(0, 1, len(all_cut_list) + 1)
        )  # +1 for the 'Combined' line
        color_cycle = cycle(colors)
        plt.figure(dpi=150)
        plt.plot(result_dict[coord], result_dict["all_cuts"], color="k", label="Combined")
        plt.fill_between(
            result_dict[coord],
            result_dict["all_cuts_lower"],
            result_dict["all_cuts_upper"],
            color="k",
            alpha=0.3,
        )
        for cut_oi in all_cut_list:
            this_color = next(color_cycle)
            plt.plot(result_dict[coord], result_dict[cut_oi], color=this_color, label=cut_oi)
            plt.fill_between(
                result_dict[coord],
                result_dict[cut_oi + "_lower"],
                result_dict[cut_oi + "_upper"],
                color=this_color,
                alpha=0.3,
            )
        plt.xlabel(coord + coord_units[coord])
        plt.ylabel("Measured Acceptance")
        plt.grid()
        plt.legend(
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=2,
            fontsize="small",
            frameon=False,
        )
        plt.title(title)

    return result_dict


def compare_2d(
    events0,
    events1,
    bins,
    title,
    xlim,
    ylim,
    label0,
    label1,
    xlabel,
    ylabel,
    coords=["z", "s2_range_50p_area"],
):
    """Compare 2D distributions of two datasets.

    :param events0: events from the first dataset
    :param events1: events from the second dataset
    :param bins: bins for the x-axis
    :param title: title of the plot
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param label0: label for the first dataset
    :param label1: label for the second dataset
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param coords: coordinates to be compared, default to ['z', 's2_range_50p_area']

    """
    events0_med = []
    events1_med = []
    events0_1sig_u = []
    events1_1sig_u = []
    events0_1sig_l = []
    events1_1sig_l = []
    events0_2sig_u = []
    events1_2sig_u = []
    events0_2sig_l = []
    events1_2sig_l = []

    for i in range(len(bins) - 1):
        mask0_i = (events0[coords[0]] >= bins[i]) & (events0[coords[0]] < bins[i + 1])
        events0_i = events0[mask0_i]
        events0_med.append(np.median(events0_i[coords[1]]))
        events0_1sig_l.append(np.percentile(events0_i[coords[1]], 16.5))
        events0_1sig_u.append(np.percentile(events0_i[coords[1]], 83.5))
        events0_2sig_l.append(np.percentile(events0_i[coords[1]], 2.5))
        events0_2sig_u.append(np.percentile(events0_i[coords[1]], 97.5))

        mask1_i = (events1[coords[0]] >= bins[i]) & (events1[coords[0]] < bins[i + 1])
        events1_i = events1[mask1_i]
        events1_med.append(np.median(events1_i[coords[1]]))
        events1_1sig_l.append(np.percentile(events1_i[coords[1]], 16.5))
        events1_1sig_u.append(np.percentile(events1_i[coords[1]], 83.5))
        events1_2sig_l.append(np.percentile(events1_i[coords[1]], 2.5))
        events1_2sig_u.append(np.percentile(events1_i[coords[1]], 97.5))

    bins_mid = (bins[:-1] + bins[1:]) / 2

    plt.figure(dpi=150)

    plt.scatter(events0[coords[0]], events0[coords[1]], s=0.5, alpha=0.2)
    plt.scatter(events1[coords[0]], events1[coords[1]], s=0.5, alpha=0.2)

    plt.plot(bins_mid, events0_med, color="tab:blue", label=label0)
    plt.plot(bins_mid, events1_med, color="tab:red", label=label1)
    plt.plot(bins_mid, events0_1sig_l, color="tab:blue", linestyle="dashed")
    plt.plot(bins_mid, events0_1sig_u, color="tab:blue", linestyle="dashed")
    plt.plot(bins_mid, events0_2sig_l, color="tab:blue", linestyle="dashed", alpha=0.5)
    plt.plot(bins_mid, events0_2sig_u, color="tab:blue", linestyle="dashed", alpha=0.5)
    plt.plot(bins_mid, events1_1sig_l, color="tab:red", linestyle="dashed")
    plt.plot(bins_mid, events1_1sig_u, color="tab:red", linestyle="dashed")
    plt.plot(bins_mid, events1_2sig_l, color="tab:red", linestyle="dashed", alpha=0.5)
    plt.plot(bins_mid, events1_2sig_u, color="tab:red", linestyle="dashed", alpha=0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.show()


def compare_bands(salt, simu, title, coords=["z", "s2_range_50p_area"], n_bins=16):
    """Wrapped comparison of 2D distributions of two datasets.

    :param salt: events from the first dataset
    :param simu: events from the second dataset
    :param title: title of the plot
    :param coords: coordinates to be compared, default to ['z', 's2_range_50p_area'], can choose
        from ['z', 's1_area', 's2_area', 's1_range_50p_area', 's1_range_90p_area', 's1_rise_time',
        's2_range_50p_area', 's2_range_90p_area']
    :param n_bins: number of bins for each coordinate, default to 16

    """
    BINS = {
        "z": np.linspace(-134, -13, n_bins),
        "s1_area": np.linspace(0, 100, n_bins),
        "s2_area": np.linspace(500, 7000, n_bins),
        "s1_range_50p_area": np.linspace(0, 300, n_bins),
        "s1_range_90p_area": np.linspace(0, 1000, n_bins),
        "s1_rise_time": np.linspace(0, 150, n_bins),
        "s2_range_50p_area": np.linspace(0, 1.5e4, n_bins),
        "s2_range_90p_area": np.linspace(0, 4e4, n_bins),
    }
    UNITS_DICT = {
        "z": "[cm]",
        "s1_area": "[PE]",
        "s2_area": "[PE]",
        "s1_range_50p_area": "[ns]",
        "s1_range_90p_area": "[ns]",
        "s1_rise_time": "[ns]",
        "s2_range_50p_area": "[ns]",
        "s2_range_90p_area": "[ns]",
    }
    bins = BINS
    units_dict = UNITS_DICT

    compare_2d(
        salt,
        simu,
        bins=bins[coords[0]],
        title=title,
        xlim=(bins[coords[0]][0], bins[coords[0]][-1]),
        ylim=(bins[coords[1]][0], bins[coords[1]][-1]),
        label0="Sprinkled",
        label1="Simulated",
        xlabel=coords[0] + units_dict[coords[0]],
        ylabel=coords[1] + units_dict[coords[1]],
        coords=coords,
    )


def show_area_bias(
    salt,
    simu,
    title,
    fraction=False,
    coord="s1_area",
    s1s2="s1",
    n_bins=16,
    ylim=(
        -5,
        20,
    ),
    bin_range=None,
):
    """Show the bias due to ambience interference VS a coordinate.

    :param salt: events from the first dataset
    :param simu: events from the second dataset
    :param title: title of the plot
    :param fraction: whether to show the bias in fraction, default to False
    :param coord: coordinate to be compared, default to 's1_area', can choose from ['z', 's1_area',
        's2_area', 's1_range_50p_area', 's1_range_90p_area', 's1_rise_time', 's2_range_50p_area',
        's2_range_90p_area']
    :param s1s2: s1 or s2, default to 's1'
    :param n_bins: number of bins for the coordinate, default to 16
    :param ylim: y-axis limits, default to (-5,20)
    :param bin_range: range of the bins, default to None

    """
    BINS = {
        "z": np.linspace(-134, -13, n_bins),
        "s1_area": np.linspace(0, 100, n_bins),
        "s2_area": np.linspace(200, 7000, n_bins),
        "s1_range_50p_area": np.linspace(0, 300, n_bins),
        "s1_range_90p_area": np.linspace(0, 1000, n_bins),
        "s1_rise_time": np.linspace(0, 150, n_bins),
        "s2_range_50p_area": np.linspace(0, 1.5e4, n_bins),
        "s2_range_90p_area": np.linspace(0, 4e4, n_bins),
    }
    UNITS_DICT = {
        "z": "[cm]",
        "s1_area": "[PE]",
        "s2_area": "[PE]",
        "s1_range_50p_area": "[ns]",
        "s1_range_90p_area": "[ns]",
        "s1_rise_time": "[ns]",
        "s2_range_50p_area": "[ns]",
        "s2_range_90p_area": "[ns]",
    }
    if bin_range is not None:
        bins = np.linspace(bin_range[0], bin_range[1], n_bins)
    else:
        bins = BINS[coord]
    units_dict = UNITS_DICT
    if s1s2 == "s1":
        bias = salt["s1_area"] - simu["s1_area"]
    elif s1s2 == "s2":
        bias = salt["s2_area"] - simu["s2_area"]
    else:
        raise ValueError

    bins_mid = (bins[1:] + bins[:-1]) / 2
    bias_med = []
    bias_1sig_u = []
    bias_1sig_l = []
    bias_2sig_u = []
    bias_2sig_l = []

    if not fraction:
        for i in range(n_bins - 1):
            mask0_i = (simu[coord] >= bins[i]) & (simu[coord] < bins[i + 1])
            bias_i = bias[mask0_i]
            bias_med.append(np.median(bias_i))
            bias_1sig_l.append(np.percentile(bias_i, 16.5))
            bias_1sig_u.append(np.percentile(bias_i, 83.5))
            bias_2sig_l.append(np.percentile(bias_i, 2.5))
            bias_2sig_u.append(np.percentile(bias_i, 97.5))
    else:
        for i in range(n_bins - 1):
            mask0_i = (simu[coord] >= bins[i]) & (simu[coord] < bins[i + 1])
            bias_i = bias[mask0_i] / simu[coord][mask0_i]
            bias_med.append(np.median(bias_i) * 100)
            bias_1sig_l.append(np.percentile(bias_i, 16.5) * 100)
            bias_1sig_u.append(np.percentile(bias_i, 83.5) * 100)
            bias_2sig_l.append(np.percentile(bias_i, 2.5) * 100)
            bias_2sig_u.append(np.percentile(bias_i, 97.5) * 100)

    plt.figure(dpi=150)
    if not fraction:
        plt.scatter(simu[coord], bias, s=0.5, alpha=0.2, color="k")
    else:
        plt.scatter(simu[coord], bias / simu[coord] * 100, s=0.5, alpha=0.2, color="k")
    plt.plot(bins_mid, bias_med, color="tab:blue", label="Median")
    plt.plot(bins_mid, bias_1sig_l, color="tab:blue", linestyle="dashed", label="1Sig")
    plt.plot(bins_mid, bias_1sig_u, color="tab:blue", linestyle="dashed")
    plt.plot(bins_mid, bias_2sig_l, color="tab:blue", linestyle="dashed", alpha=0.5, label="2Sig")
    plt.plot(bins_mid, bias_2sig_u, color="tab:blue", linestyle="dashed", alpha=0.5)
    if not fraction:
        plt.ylabel(f"Change in {s1s2} Area [PE]")
    else:
        plt.ylabel(f"Change in {s1s2} Area [%%]")
    plt.xlabel(coord + units_dict[coord])
    plt.xlim(bins[0], bins[-1])
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

    result_dict = {}
    result_dict["coord"] = coord
    result_dict["s1s2"] = s1s2
    result_dict["bins_mid"] = bins_mid
    result_dict["bias_med"] = bias_med
    result_dict["bias_1sig_u"] = bias_1sig_u
    result_dict["bias_1sig_l"] = bias_1sig_l
    result_dict["bias_2sig_u"] = bias_2sig_u
    result_dict["bias_2sig_l"] = bias_2sig_l

    return result_dict


def show_eff2d(
    events,
    events_selected,
    coord=("s1_area", "s2_area"),
    bins=(np.linspace(0, 100, 101), np.linspace(500, 7000, 101)),
    title="Matching Acceptance",
    vmin_vmax=(0, 1),  # New parameter to set color bar range
    min_counts=100,
):
    """Show the acceptance in 2D coordinates.

    :param events: events before some selection
    :param events_selected: events after some selection
    :param coord: coordinates to be compared, default to ('s1_area', 's2_area')
    :param bins: bins for the coordinates, default to (np.linspace(0,100,101),
        np.linspace(500,7000,101))
    :param title: title of the plot, default to "Matching Acceptance"
    :param vmin_vmax: range of color bar, default to (0,1)
    :param min_counts: minimum number of counts in a bin to be considered, default to 100
    :return: efficiency, xedges, yedges

    """
    label_dict = {
        "e_ces": "Simulated CES [keV]",
        "s1_area": "Simulated S1 Area [PE]",
        "s2_area": "Simulated S2 Area [PE]",
        "z": "Z [cm]",
    }

    # Count the number of events in each bin
    counts, xedges, yedges = np.histogram2d(events[coord[0]], events[coord[1]], bins=bins)
    counts_selected, xedges, yedges = np.histogram2d(
        events_selected[coord[0]], events_selected[coord[1]], bins=bins
    )

    # Compute efficiency
    eff = counts_selected / counts
    eff[np.isnan(eff)] = 0
    eff[counts < min_counts] = 0

    # Plot
    plt.figure(dpi=150)
    plt.imshow(
        eff.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="viridis",
        vmin=vmin_vmax[0],  # Set minimum value for color scale
        vmax=vmin_vmax[1],  # Set maximum value for color scale
    )
    plt.colorbar(label="Efficiency")
    plt.xlabel(label_dict[coord[0]])
    plt.ylabel(label_dict[coord[1]])
    plt.title(title)
    plt.show()
    return eff, xedges, yedges


def show_eff1d(
    events_simu,
    events_simu_matched_to_salt,
    mask_salt_cut=None,
    coord="e_ces",
    bins=np.linspace(0, 12, 25),
    labels_hist=[
        "Simulation before matching&cuts",
        "Simulation after matching",
        "Simulation after matching&cuts",
    ],
    labels_eff=["Matching", "Cut (Already Matched)"],
    title="Matching Acceptance and Cut Acceptance",
):
    """Show the acceptance of matching and cuts in 1D coordinates.

    :param events_simu: events from the simulated dataset
    :param events_simu_matched_to_salt: events from the simulated dataset matched to sprinkled
    :param mask_salt_cut: mask of the sprinkled dataset with cuts, default to None
    :param coord: coordinate to be compared, default to 'e_ces', can choose from ['e_ces',
        's1_area', 's2_area']
    :param bins: bins for the coordinate, default to np.linspace(0,12,25)
    :param title: title of the plot, default to "Matching Acceptance and Cut Acceptance"

    """
    xlabel_dict = {
        "e_ces": "Simulated CES [keV]",
        "s1_area": "Simulated S1 Area [PE]",
        "s2_area": "Simulated S2 Area [PE]",
        "z": "Z [cm]",
    }

    # Histogram
    plt.figure(dpi=150)
    plt.hist(events_simu[coord], bins=bins, label=labels_hist[0])
    plt.hist(events_simu_matched_to_salt[coord], bins=bins, label=labels_hist[1])
    if mask_salt_cut is not None:
        plt.hist(
            events_simu_matched_to_salt[mask_salt_cut][coord],
            bins=bins,
            color="tab:red",
            label=labels_hist[2],
        )
    plt.yscale("log")
    plt.xlabel(xlabel_dict[coord])
    plt.legend()
    plt.ylabel("Counts")
    plt.title(title)
    plt.show()

    # Efficiency curves with Clopper-Pearson uncertainty estimation
    plt.figure(dpi=150)
    counts_events_simu, bins = np.histogram(events_simu[coord], bins=bins)
    counts_events_simu_matched_to_salt, bins = np.histogram(
        events_simu_matched_to_salt[coord], bins=bins
    )
    if mask_salt_cut is not None:
        counts_events_simu_matched_to_salt_after_cuts, bins = np.histogram(
            events_simu_matched_to_salt[mask_salt_cut][coord], bins=bins
        )
    coords = (bins[1:] + bins[:-1]) / 2

    # Get Clopper-Pearson uncertainty
    matching_u = []
    matching_l = []
    cuts_u = []
    cuts_l = []
    for i in range(len(coords)):
        matching_interval = binomtest(
            counts_events_simu_matched_to_salt[i], counts_events_simu[i]
        ).proportion_ci()
        matching_l.append(matching_interval.low)
        matching_u.append(matching_interval.high)
        if mask_salt_cut is not None:
            cuts_interval = binomtest(
                counts_events_simu_matched_to_salt_after_cuts[i],
                counts_events_simu_matched_to_salt[i],
            ).proportion_ci()
            cuts_l.append(cuts_interval.low)
            cuts_u.append(cuts_interval.high)
    matching_u = np.array(matching_u)
    matching_l = np.array(matching_l)
    if mask_salt_cut is not None:
        cuts_u = np.array(cuts_u)
        cuts_l = np.array(cuts_l)

    plt.plot(
        coords,
        counts_events_simu_matched_to_salt / counts_events_simu,
        label=labels_eff[0],
        color="tab:blue",
    )
    plt.fill_between(coords, matching_l, matching_u, alpha=0.5, color="tab:blue")
    if mask_salt_cut is not None:
        plt.plot(
            coords,
            counts_events_simu_matched_to_salt_after_cuts / counts_events_simu_matched_to_salt,
            label=labels_eff[1],
            color="tab:orange",
        )
        plt.fill_between(coords, cuts_l, cuts_u, alpha=0.5, color="tab:orange")

    plt.xlabel(xlabel_dict[coord])
    plt.legend()
    plt.ylabel("Acceptance")
    plt.title(title)
    plt.show()


def apply_peaks_daq_cuts(st_data, runs, peaks, proximity_extension=int(0.25e6)):
    """
    Analogy to DAQVeto in cutax: https://github.com/XENONnT/cutax/blob/fb9c23cea86b44c0402437189fc606399d4e134c/cutax/cuts/daq_veto.py#L8  # noqa
    Apply cuts based on veto_intervals, using strax.touching_windows
    :param st_data: context for data in cutax
    :param runs: ordered runs list
    :param peaks: peaks level data with ordered times
    :param proximity_extension: extension of the veto proximity cut, default to int(0.25e6)
    :return: mask_daq_cut mask for veto cuts
    """
    # Load veto_intervals
    veto_intervals = st_data.get_array(runs, "veto_intervals")

    mask_daq_cut = np.ones(len(peaks), dtype=bool)
    # Once the peaks are in proximity of the veto intervals, they are cut
    windows = strax.touching_windows(veto_intervals, peaks, window=proximity_extension)
    windows_length = windows[:, 1] - windows[:, 0]
    mask_daq_cut[windows_length > 0] = False

    return mask_daq_cut
