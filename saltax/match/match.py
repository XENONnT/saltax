import numpy as np
from tqdm import tqdm


def filter_out_missing_s1_s2(truth, match):
    """
    Filter out simulated events that have no S1 or S2 or both
    :param truth: truth from wfsim
    :param match: match_acceptance_extended from pema 
    :return: filtered truth and match
    """
    bad_mask = np.zeros(len(truth), dtype=bool)
    max_event_number = np.max(np.unique(truth['event_number']))

    for event_id in np.arange(max_event_number)+1:
        selected_truth = truth[truth['event_number']==event_id]
        indices = np.where(truth['event_number']==event_id)

        # If cannot find any S1 or S2, then remove the event
        if len(selected_truth[selected_truth['type']==1]) == 0:
            bad_mask[indices] = True
        elif len(selected_truth[selected_truth['type']==2]) == 0:
            bad_mask[indices] = True

        # If any S1 or S2 has 0 photon or 0 electron, then remove the event
        elif 0 in selected_truth[selected_truth['type']==1]['n_photon']:
            bad_mask[indices] = True
        elif 0 in selected_truth[selected_truth['type']==2]['n_electron']:
            bad_mask[indices] = True

    print("Filter out %s percent of events due to missing S1 or S2 or both"\
           % (np.sum(bad_mask)/len(truth)*100))
    return truth[~bad_mask], match[~bad_mask]


def filter_out_multiple_s1_s2(truth, match):
    """
    Filter out simulated events that have multiple S1 or S2 reconstructed from wfsim.
    :param truth: truth from wfsim
    :param match: match_acceptance_extended from pema
    :return: filtered truth and match
    """
    bad_mask = np.zeros(len(truth), dtype=bool)
    max_event_number = np.max(np.unique(truth['event_number']))
    
    for event_id in np.arange(max_event_number)+1:
        selected_truth = truth[truth['event_number']==event_id]
        indices = np.where(truth['event_number']==event_id)

        # If there are multiple S1 or S2, then remove the event
        if (len(selected_truth[selected_truth['type']==1]) != 1 or 
            len(selected_truth[selected_truth['type']==2]) != 1):
            bad_mask[indices] = True

    print("Filter out %s percent of events due to multiple S1 or S2"\
           % (np.sum(bad_mask)/len(truth)*100))
    return truth[~bad_mask], match[~bad_mask]


def filter_out_not_found(truth, match):
    """
    Filter out simulated events whose S1 and S2 are not found by pema.
    :param truth: truth from wfsim
    :param match: match_acceptance_extended from pema
    :return: filtered truth and match
    """
    bad_mask = np.zeros(len(truth), dtype=bool)
    max_event_number = np.max(np.unique(truth['event_number']))
    
    for event_id in np.arange(max_event_number)+1:
        selected_truth = truth[truth['event_number']==event_id]
        indices = np.where(truth['event_number']==event_id)
        selected_match = match[indices]
        
        # The only outcome should be "found", otherwise remove the event
        if len(selected_match['outcome']) == 1:
            if np.unique(selected_match['outcome'])[0] != "found":
                bad_mask[indices] = True
        else:
            bad_mask[indices] = True

    print("Filter out %s percent of events due to S1 or S2 not found"\
           % (np.sum(bad_mask)/len(truth)*100))
    return truth[~bad_mask], match[~bad_mask]


def pair_events_to_filtered_truth(truth, events):
    """
    Pair events to filtered truth.
    :param truth: filtered truth
    :param events: events from wfsim or saltax after reconstruction
    :return: matched_to, the index of matched truth event for each event
    """
    array_dtype = [
        ('s1_time', np.int64),
        ('s1_endtime', np.int64),
        ('s2_time', np.int64),
        ('s2_endtime', np.int64),
    ]
    matched_truth_events_timing = np.zeros(int(len(truth)/2), dtype=array_dtype)
    matched_truth_events_timing['s1_time'] = truth[truth['type']==1]['time']
    matched_truth_events_timing['s1_endtime'] = truth[truth['type']==1]['endtime']
    matched_truth_events_timing['s2_time'] = truth[truth['type']==2]['time']
    matched_truth_events_timing['s2_endtime'] = truth[truth['type']==2]['endtime']

    matched_to = np.zeros(len(matched_truth_events_timing), dtype=int)
    for i,e_simu in enumerate(tqdm(matched_truth_events_timing)):
        # Find the events whose S1 and S2 overlap with the truth's S1 and S2 time ranges
        # Note that we only care about main S1 and main S2, otherwise we consider lost
        # Temporary S1 only
        j_selected_events = np.where((events['s1_endtime']>=e_simu['s1_time'])&
                                     (e_simu['s1_endtime']>=events['s1_time']))[0]
        #j_selected_events = np.where((events['s1_endtime']>=e_simu['s1_time'])&
        #                             (e_simu['s1_endtime']>=events['s1_time'])&
        #                             (events['s2_endtime']>=e_simu['s2_time'])&
        #                             (e_simu['s2_endtime']>=events['s2_time']))[0]
        assert len(j_selected_events) <= 1, "Multiple events found for one truth event!?"
        
        # If no event is found, then we consider lost
        if len(j_selected_events) == 0:
            matched_to[i] = -99999
        # If only one event is found, then we consider matched
        elif len(j_selected_events) == 1:
            matched_to[i] = j_selected_events[0]

    return matched_to


def pair_events_to_matched_simu(matched_simu, events):
    """
    Pair salted events to matched simulation events.
    :param matched_simu: simulation events already matched to truth
    :param events: events from saltax after reconstruction
    :return: matched_to, the index of matched simulation events for each event
    """
    matched_to = np.zeros(len(matched_simu), dtype=int)
    for i,e_simu in enumerate(tqdm(matched_simu)):
        # Find the events whose S1 and S2 overlap with the truth's S1 and S2 time ranges
        # Note that we only care about main S1 and main S2, otherwise we consider lost
        j_selected_events = np.where((events['s1_endtime']>=e_simu['s1_time'])&
                                     (e_simu['s1_endtime']>=events['s1_time']))[0]
        
        j_selected_events = np.where((events['s1_endtime']>=e_simu['s1_time'])&
                                     (e_simu['s1_endtime']>=events['s1_time'])&
                                     (events['s2_endtime']>=e_simu['s2_time'])&
                                     (e_simu['s2_endtime']>=events['s2_time']))[0]
        assert len(j_selected_events) <= 1, "Multiple events found for one truth event!?"
        
        # If no event is found, then we consider lost
        if len(j_selected_events) == 0:
            matched_to[i] = -99999
        # If only one event is found, then we consider matched
        elif len(j_selected_events) == 1:
            matched_to[i] = j_selected_events[0]

    return matched_to


def pair_salt_to_simu(truth, match, events_simu, events_salt):
    """
    Filter out bad simulation truth and then pair salted events to matched simulation events.
    :param truth: filtered truth
    :param match: match_acceptance_extended from pema
    :param events_simu: events from wfsim
    :param events_salt: events from saltax after reconstruction
    :return: ind_salt_matched_to_simu, ind_simu_matched_to_truth, truth_filtered, match_filtered
    """
    truth_filtered, match_filtered = filter_out_missing_s1_s2(truth, match)
    truth_filtered, match_filtered = filter_out_multiple_s1_s2(truth_filtered, match_filtered)
    # Temporarily turn off this filter because of wfsim bug in s2 timing
    #truth_filtered, match_filtered = filter_out_not_found(truth_filtered, match_filtered) 

    ind_simu_matched_to_truth = pair_events_to_filtered_truth(truth_filtered, events_simu)
    events_simu_matched_to_truth = events_simu[ind_simu_matched_to_truth[ind_simu_matched_to_truth>=0]]

    ind_salt_matched_to_simu = pair_events_to_matched_simu(events_simu_matched_to_truth, 
                                                           events_salt)
    
    return ind_salt_matched_to_simu, ind_simu_matched_to_truth, truth_filtered, match_filtered


def match(truth, match, events_simu, events_salt):
    """
    Match salted events to simulation events.
    :param truth: truth from wfsim
    :param match: match_acceptance_extended from pema
    :param events_simu: event_info from wfsim
    :param events_salt: event_info from saltax
    :return: events_salt_matched_to_simu, events_simu_matched_to_salt: matched events with equal length
    """
    ind_salt_matched_to_simu, \
        ind_simu_matched_to_truth, \
            _, _ = pair_salt_to_simu(truth, match, events_simu, events_salt)

    events_simu_matched_to_truth = events_simu[ind_simu_matched_to_truth[ind_simu_matched_to_truth>=0]]
    events_salt_matched_to_simu = events_salt[ind_salt_matched_to_simu[ind_salt_matched_to_simu>=0]]
    events_simu_matched_to_salt = events_simu_matched_to_truth[ind_salt_matched_to_simu>=0]

    return events_salt_matched_to_simu, events_simu_matched_to_salt
