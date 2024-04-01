import numpy as np
from tqdm import tqdm
import strax


def match_events(events_simu, events_salt,
                 event_window_fuzz=0, s1_window_fuzz=100, s2_window_fuzz=0):
    """
    Match salted events to simulation events based on event or S1-S2 timing, without checking 
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
        6. Find indcies of events_salt whose S2 time range overlaps with events_simu_filtered's S2 time 
           range: ind_salt_s2_found. Those events_salt[ind_salt_s2_found] are called events_salt_s2_found.
        7. Find indcies of events_simu_filtered whose S2 time range overlaps with events_salt_s2_found:
           ind_simu_s2_found. Those events_simu_filtered[ind_simu_s2_found] are called events_simu_s2_found.
    :param events_simu: event_info from wfsim
    :param events_salt: event_info from saltax
    :param event_window_fuzz: extended time range to consider as matched for events, default 0 ns
    :param s1_window_fuzz: extended time range to consider as matched for S1, default 100 ns. Reference 
                           https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt_sr1:ambe_selection
    :param s2_window_fuzz: extended time range to consider as matched for S2, default 0 ns
    :return: events_simu_filtered, 
             ind_salt_event_found, ind_simu_event_found, ind_simu_event_lost, ind_simu_event_split,
             ind_salt_s1_found, ind_simu_s1_found, 
             ind_salt_s2_found, ind_simu_s2_found
    """
    # Step 1.
    # Filter out events_simu which is missing S1
    events_simu_filtered = events_simu[events_simu['s1_time']>=0]

    # Step 2 & 3.
    # Find indices of events_salt whose time range overlaps with events_simu_filtered
    event_touching_windows = strax.touching_windows(events_salt, events_simu_filtered, 
                                                    window=event_window_fuzz)
    event_windows_length = event_touching_windows[:,1] - event_touching_windows[:,0]
    # If windows_length is not 1, the sprinkled event is either split or not found
    mask_simu_event_found = event_windows_length == 1
    mask_simu_event_lost  = event_windows_length == 0
    mask_simu_event_split = event_windows_length >= 2
    ind_simu_event_found = np.where(mask_simu_event_found)[0]
    ind_simu_event_lost  = np.where(mask_simu_event_lost)[0]
    ind_simu_event_split = np.where(mask_simu_event_split)[0]
    # Find indcies of events_salt whose event time range overlaps with events_simu_filtered
    ind_salt_event_found = event_touching_windows[mask_simu_event_found][:,0]

    # Step 4 & 5.
    # Assumed s1 times has been sorted! It should be a safe assumption because their events are sorted.
    s1_touching_windows = strax.processing.general._touching_windows(
        events_salt["s1_time"],
        events_salt["s1_endtime"],
        events_simu_filtered["s1_time"],
        events_simu_filtered["s1_endtime"],
        window=s1_window_fuzz
    )
    s1_windows_length = s1_touching_windows[:,1] - s1_touching_windows[:,0]
    mask_simu_s1_found = s1_windows_length == 1
    ind_simu_s1_found = np.where(mask_simu_s1_found)[0]
    ind_salt_s1_found = s1_touching_windows[mask_simu_s1_found][:,0]

    # Step 6 & 7.
    # Assumed s2 times has been sorted! It should be a safe assumption because their events are sorted.
    s2_touching_windows = strax.processing.general._touching_windows(
        events_salt["s2_time"],
        events_salt["s2_endtime"],
        events_simu_filtered["s2_time"],
        events_simu_filtered["s2_endtime"],
        window=s2_window_fuzz
    )
    s2_windows_length = s2_touching_windows[:,1] - s2_touching_windows[:,0]
    mask_simu_s2_found = s2_windows_length == 1
    ind_simu_s2_found = np.where(mask_simu_s2_found)[0]
    ind_salt_s2_found = s2_touching_windows[mask_simu_s2_found][:,0]

    return (
        events_simu_filtered,
        ind_salt_event_found, ind_simu_event_found, ind_simu_event_lost, ind_simu_event_split,
        ind_salt_s1_found, ind_simu_s1_found,
        ind_salt_s2_found, ind_simu_s2_found
    )


def match_peaks(truth, match, peaks_simu, peaks_salt):
    """
    Match salted peaks to simulation peaks.
    :param truth: truth from wfsim
    :param match: match_acceptance_extended from pema
    :param peaks_simu: peaks from wfsim
    :param peaks_salt: peaks from saltax
    :param type: 1 for S1, 2 for S2 to require 'found'
    :return: peaks_salt_matched_to_simu, peaks_simu_matched_to_salt: matched peaks with equal length
    """
    # Remove bad simulation truth and then their paired simulated events
    truth = truth[match['matched_to']>=0]
    match = match[match['matched_to']>=0]

    ind_salt_matched_to_simu, \
        ind_simu_matched_to_truth = pair_salt_to_simu_peaks(match, peaks_simu, peaks_salt)

    peaks_simu_matched_to_truth = peaks_simu[ind_simu_matched_to_truth[ind_simu_matched_to_truth>=0]]
    peaks_salt_matched_to_simu = peaks_salt[ind_salt_matched_to_simu[ind_salt_matched_to_simu>=0]]
    peaks_simu_matched_to_salt = peaks_simu_matched_to_truth[ind_salt_matched_to_simu>=0]

    return peaks_salt_matched_to_simu, peaks_simu_matched_to_salt


def pair_salt_to_simu_peaks(match, peaks_simu, peaks_salt):
    """
    Filter out bad simulation truth and then pair salted events to matched simulation events.
    :param match: match_acceptance_extended from pema
    :param peaks_simu: peaks from wfsim
    :param peaks_salt: peaks from saltax after reconstruction
    :return: ind_salt_matched_to_simu, ind_simu_matched_to_truth
    """
    ind_simu_matched_to_truth = match['matched_to']
    peaks_simu_matched_to_truth = peaks_simu[ind_simu_matched_to_truth]

    (_ind_simu_matched_to_truth, 
     ind_salt_matched_to_simu) = pair_peaks_to_matched_simu(peaks_simu_matched_to_truth, 
                                                            peaks_salt)
    ind_simu_matched_to_truth = ind_simu_matched_to_truth[_ind_simu_matched_to_truth]

    return ind_salt_matched_to_simu, ind_simu_matched_to_truth


def pair_peaks_to_matched_simu(matched_simu, peaks, safeguard=0):
    """
    Pair salted peaks to simulation peaks who have been matched to truth.
    :param matched_simu: simulation peaks already matched to truth
    :param peaks: peaks from saltax after reconstruction
    :param safeguard: extension of time range to consider as matched, as a workaround for timing problem in wfsim s2
    :return: ind_simu_matched_to_salt: the index of matched simulation peaks for each salted peak
    :return: ind_salt_matched_to_simu: the index of matched salted peaks for each simulation peak
    """
    windows = strax.touching_windows(peaks, 
                                     matched_simu, 
                                     window=safeguard)
    windows_length = windows[:,1] - windows[:,0]
    simu_matched_to_salt_mask = windows_length == 1
    print("Filter out %s percent of simulation due to multiple or no matched sprinkled \
           peaks"%(np.sum(~simu_matched_to_salt_mask)/len(matched_simu)*100))

    matched_simu = matched_simu[simu_matched_to_salt_mask]
    ind_simu_matched_to_salt = np.where(simu_matched_to_salt_mask)[0]
    ind_salt_matched_to_simu = windows[simu_matched_to_salt_mask][:,0]

    return ind_simu_matched_to_salt, ind_salt_matched_to_simu
