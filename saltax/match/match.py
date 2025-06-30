import numpy as np
import strax


def match_events(
    events_simu, events_salt, event_window_fuzz=0, s1_window_fuzz=100, s2_window_fuzz=0
):
    """Match salted events to simulation events based on event or S1-S2 timing, without checking
    simulation truth.

    The procedures are:
        1. Filter out events_simu which is missing S1. Then the rest events_simu is like 'truth'
           without ambience interference. Those events_simu are called events_simu_filtered.
        2. Find indices of events_salt whose time range overlaps with events_simu_filtered:
           ind_salt_event_found. Those events_salt[ind_salt_event_found] are called
           events_salt_event_found.
        3. Find indcies of events_simu_filtered whose time range overlaps with events_salt_event_found:
           ind_simu_event_found. Those events_simu_filtered[ind_simu_event_found] are called
           events_simu_event_found.
        4. Find indcies of events_salt whose S1 time range overlaps with events_simu_filtered's S1 time
           range: ind_salt_s1_found. Those events_salt[ind_salt_s1_found] are called events_salt_s1_found.
        5. Find indcies of events_simu_filtered whose S1 time range overlaps with events_salt_s1_found:
           ind_simu_s1_found. Those events_simu_filtered[ind_simu_s1_found] are called events_simu_s1_found.
           The processes in step 4 and 5 are repeated also for alt_s1.
        6. Find indcies of events_salt whose S2 time range overlaps with events_simu_filtered's S2 time
           range: ind_salt_s2_found. Those events_salt[ind_salt_s2_found] are called events_salt_s2_found.
        7. Find indcies of events_simu_filtered whose S2 time range overlaps with events_salt_s2_found:
           ind_simu_s2_found. Those events_simu_filtered[ind_simu_s2_found] are called events_simu_s2_found.
           The processes in step 6 and 7 are repeated also for alt_s2.
    :param events_simu: event_info from fuse
    :param events_salt: event_info from saltax
    :param event_window_fuzz: extended time range to consider as matched for events, default 0 ns
    :param s1_window_fuzz: extended time range to consider as matched for S1, default 100 ns. Reference
                           https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt_sr1:ambe_selection
    :param s2_window_fuzz: extended time range to consider as matched for S2, default 0 ns
    :return: events_simu_filtered,
             ind_salt_event_found, ind_simu_event_found, ind_simu_event_lost, ind_simu_event_split,
             ind_salt_s1_found, ind_simu_s1_found, ind_salt_s1_made_alt, ind_simu_s1_made_alt,
             ind_salt_s2_found, ind_simu_s2_found, ind_salt_s2_made_alt, ind_simu_s2_made_alt

    """
    # Step 1.
    # Filter out events_simu which is missing S1
    events_simu_filtered = events_simu[events_simu["s1_time"] >= 0]

    # Step 2 & 3.
    # Find indices of events_salt whose time range overlaps with events_simu_filtered
    event_touching_windows = strax.touching_windows(
        events_salt, events_simu_filtered, window=event_window_fuzz
    )
    event_windows_length = event_touching_windows[:, 1] - event_touching_windows[:, 0]
    # If windows_length is not 1, the sprinkled event is either split or not found
    mask_simu_event_found = event_windows_length == 1
    mask_simu_event_lost = event_windows_length == 0
    mask_simu_event_split = event_windows_length >= 2
    ind_simu_event_found = np.where(mask_simu_event_found)[0]
    ind_simu_event_lost = np.where(mask_simu_event_lost)[0]
    ind_simu_event_split = np.where(mask_simu_event_split)[0]
    # Find indcies of events_salt whose event time range overlaps with events_simu_filtered
    ind_salt_event_found = event_touching_windows[mask_simu_event_found][:, 0]

    # Step 4 & 5.
    # Assumed s1 times has been sorted! It should be a safe assumption because their events are sorted.
    s1_touching_windows = strax.processing.general._touching_windows(
        events_salt["s1_time"],
        events_salt["s1_endtime"],
        events_simu_filtered["s1_time"],
        events_simu_filtered["s1_endtime"],
        window=s1_window_fuzz,
    )
    s1_windows_length = s1_touching_windows[:, 1] - s1_touching_windows[:, 0]
    mask_simu_s1_found = s1_windows_length == 1
    ind_simu_s1_found = np.where(mask_simu_s1_found)[0]
    ind_salt_s1_found = s1_touching_windows[mask_simu_s1_found][:, 0]
    # Repeat the process for alt_s1
    alt_s1_touching_windows = strax.processing.general._touching_windows(
        events_salt["alt_s1_time"],
        events_salt["alt_s1_endtime"],
        events_simu_filtered["s1_time"],
        events_simu_filtered["s1_endtime"],
        window=s1_window_fuzz,
    )
    alt_s1_windows_length = alt_s1_touching_windows[:, 1] - alt_s1_touching_windows[:, 0]
    mask_simu_s1_made_alt = alt_s1_windows_length == 1
    ind_simu_s1_made_alt = np.where(mask_simu_s1_made_alt)[0]
    ind_salt_s1_made_alt = alt_s1_touching_windows[mask_simu_s1_made_alt][:, 0]

    # Step 6 & 7.
    # Assumed s2 times has been sorted! It should be a safe assumption because their events are sorted.
    s2_touching_windows = strax.processing.general._touching_windows(
        events_salt["s2_time"],
        events_salt["s2_endtime"],
        events_simu_filtered["s2_time"],
        events_simu_filtered["s2_endtime"],
        window=s2_window_fuzz,
    )
    s2_windows_length = s2_touching_windows[:, 1] - s2_touching_windows[:, 0]
    mask_simu_s2_found = s2_windows_length == 1
    ind_simu_s2_found = np.where(mask_simu_s2_found)[0]
    ind_salt_s2_found = s2_touching_windows[mask_simu_s2_found][:, 0]
    # Repeat the process for alt_s2
    alt_s2_touching_windows = strax.processing.general._touching_windows(
        events_salt["alt_s2_time"],
        events_salt["alt_s2_endtime"],
        events_simu_filtered["s2_time"],
        events_simu_filtered["s2_endtime"],
        window=s2_window_fuzz,
    )
    alt_s2_windows_length = alt_s2_touching_windows[:, 1] - alt_s2_touching_windows[:, 0]
    mask_simu_s2_made_alt = alt_s2_windows_length == 1
    ind_simu_s2_made_alt = np.where(mask_simu_s2_made_alt)[0]
    ind_salt_s2_made_alt = alt_s2_touching_windows[mask_simu_s2_made_alt][:, 0]

    return (
        events_simu_filtered,
        ind_salt_event_found,
        ind_simu_event_found,
        ind_simu_event_lost,
        ind_simu_event_split,
        ind_salt_s1_found,
        ind_simu_s1_found,
        ind_salt_s1_made_alt,
        ind_simu_s1_made_alt,
        ind_salt_s2_found,
        ind_simu_s2_found,
        ind_salt_s2_made_alt,
        ind_simu_s2_made_alt,
    )


def match_peaks(peaks_simu, peaks_salt):
    """Match salted peaks to simulation peaks based on peak timing, without checking simulation
    truth.

    The procedures are:
        1. Find indices of peaks_salt whose time range overlaps with peaks_simu: ind_salt_peak_found.
           Those peaks_salt[ind_salt_peak_found] are called peaks_salt_peak_found.
        2. Find indcies of peaks_simu whose time range overlaps with peaks_salt_peak_found:
           ind_simu_peak_found. Those peaks_simu[ind_simu_peak_found] are called peaks_simu_peak_found.
        3. If window_length is 0, the sprinkled peak is lost: ind_simu_peak_lost.
        4. If window_length is larger than 1, the sprinkled peak is split: ind_simu_peak_split.
           When a peak is split, the peak with the largest area is selected.
    :param peaks_simu: peaks from fuse, typically peak_basics
    :param peaks_salt: peaks from saltax, typically peak_basics
    :return: ind_salt_peak_found, ind_simu_peak_found,
             ind_simu_peak_lost,
             ind_salt_peak_split, ind_simu_peak_split

    """
    # Find indices of peaks_salt whose time range overlaps with peaks_simu
    peak_touching_windows = strax.touching_windows(peaks_salt, peaks_simu)
    peak_windows_length = peak_touching_windows[:, 1] - peak_touching_windows[:, 0]

    # If windows_length is not 1, the sprinkled peak is either split or not found
    mask_simu_peak_found = peak_windows_length == 1
    ind_simu_peak_found = np.where(mask_simu_peak_found)[0]
    # Find indcies of peaks_simu whose time range overlaps with peaks_salt_peak_found
    ind_salt_peak_found = peak_touching_windows[mask_simu_peak_found][:, 0]

    # If window_length is 0, the sprinkled peak is lost
    ind_simu_peak_lost = np.where(peak_windows_length == 0)[0]

    # If window_length is larger than 1, the sprinkled peak is split
    ind_simu_peak_split = np.where(peak_windows_length > 1)[0]
    # When a peak is split, the peak with the largest area is selected
    ind_salt_peak_split = []
    for i in ind_simu_peak_split:
        # get peaks_salt touched by peaks_simu
        _peaks_salt_split = peaks_salt[peak_touching_windows[i][0] : peak_touching_windows[i][1]]
        # get the index of the peak with the largest area
        relative_ind_max_area = np.argmax(_peaks_salt_split["area"])
        ind_salt_peak_split.append(relative_ind_max_area + peak_touching_windows[i][0])
    ind_salt_peak_split = np.array(ind_salt_peak_split)

    return (
        ind_salt_peak_found,
        ind_simu_peak_found,
        ind_simu_peak_lost,
        ind_salt_peak_split,
        ind_simu_peak_split,
    )
