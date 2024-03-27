import numpy as np
from tqdm import tqdm
import gc 
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from itertools import cycle
import saltax

ALL_CUTS = np.array([
            'cut_daq_veto',
            'cut_interaction_exists',
            'cut_main_is_valid_triggering_peak',
            'cut_run_boundaries',
            'cut_s1_area_fraction_top',
            'cut_s1_max_pmt',
            'cut_s1_pattern_bottom',
            'cut_s1_pattern_top',
            'cut_s1_single_scatter',
            'cut_s1_tightcoin_3fold',
            'cut_s1_width',
            'cut_s2_pattern',
            'cut_s2_recon_pos_diff',
            'cut_s2_single_scatter',
            'cut_s2_width',
            'cut_cs2_area_fraction_top',
            'cut_shadow',
            'cut_ambience',])
ALL_CUTS_EXCEPT_S2PatternS1Width = np.array([
            'cut_daq_veto',
            'cut_interaction_exists',
            'cut_main_is_valid_triggering_peak',
            'cut_run_boundaries',
            'cut_s1_area_fraction_top',
            'cut_s1_max_pmt',
            'cut_s1_pattern_bottom',
            'cut_s1_pattern_top',
            'cut_s1_single_scatter',
            'cut_s1_tightcoin_3fold',
            'cut_s2_recon_pos_diff',
            'cut_s2_single_scatter',
            'cut_s2_width',
            'cut_cs2_area_fraction_top',
            'cut_shadow',
            'cut_ambience',])
AmBe_CUTS_EXCEPT_S2PatternS1Width = np.array([
            'cut_daq_veto',
            'cut_interaction_exists',
            'cut_main_is_valid_triggering_peak',
            'cut_run_boundaries',
            'cut_s1_area_fraction_top',
            'cut_s1_max_pmt',
            'cut_s1_pattern_bottom',
            'cut_s1_pattern_top',
            'cut_s1_single_scatter',
            'cut_s1_tightcoin_3fold',
            'cut_s2_recon_pos_diff',
            'cut_s2_single_scatter',
            'cut_s2_width',
            'cut_cs2_area_fraction_top',])


def load_peaks(runs, st_salt, st_simu, 
               plugins=('peak_basics', 'peak_positions_mlp'), 
               truth_fv_cut=True):
    """
    Load peaks from the runs and do basic filtering suggeted by saltax.match_peaks
    :param runs: list of runs.
    :param st_salt: saltax context for salt mode
    :param st_simu: saltax context for simu mode
    :param plugins: plugins to be loaded, default to ('peak_basics', 'peak_positions_mlp')
    :param truth_fv_cut: whether to apply truth FV cut, default to True
    :return: peaks_simu: peaks related plugins from simulated dataset
    :return: peaks_salt:  peaks related plugins from sprinkled dataset
    :return: truth: truth information in simulation
    :return: match: pema match information
    :return: peaks_salt_matched_to_simu: peaks related plugins from sprinkled matched to simulated
    :return: peaks_simu_matched_to_salt: peaks related plugins from simulated matched to sprinkled
    """
    for i, run in enumerate(runs):
        print("Loading run %s"%(run))

        peaks_simu_i = st_simu.get_array(run, plugins, 
                                         progress_bar=False)
        peaks_salt_i = st_salt.get_array(run, plugins, 
                                         progress_bar=False)
        truth_i = st_simu.get_array(run, 'truth', progress_bar=False)
        match_i = st_simu.get_array(run, 'match_acceptance_extended', progress_bar=False)

        # TODO: Ugly hardcoding for FV cut, need to be fixed
        if truth_fv_cut:
            peaks_salt_matched_to_simu_i, \
                peaks_simu_matched_to_salt_i = saltax.match_peaks(truth_i[(truth_i['z']<-13)&(truth_i['z']>-145)&(truth_i['x']**2+truth_i['y']**2<64**2)], 
                                                                match_i[(truth_i['z']<-13)&(truth_i['z']>-145)&(truth_i['x']**2+truth_i['y']**2<64**2)],
                                                                peaks_simu_i, peaks_salt_i)    
        else:
            peaks_salt_matched_to_simu_i, \
                peaks_simu_matched_to_salt_i = saltax.match_peaks(truth_i, match_i, peaks_simu_i, peaks_salt_i)

        if i==0:
            peaks_simu = peaks_simu_i
            peaks_salt = peaks_salt_i
            truth = truth_i
            match = match_i
            peaks_salt_matched_to_simu = peaks_salt_matched_to_simu_i
            peaks_simu_matched_to_salt = peaks_simu_matched_to_salt_i
        else:
            peaks_simu = np.concatenate((peaks_simu,peaks_simu_i))
            peaks_salt = np.concatenate((peaks_salt,peaks_salt_i))
            truth = np.concatenate((truth,truth_i))
            match = np.concatenate((match,match_i))
            peaks_salt_matched_to_simu = np.concatenate((peaks_salt_matched_to_simu,
                                                      peaks_salt_matched_to_simu_i))
            peaks_simu_matched_to_salt = np.concatenate((peaks_simu_matched_to_salt,
                                                      peaks_simu_matched_to_salt_i))

    return peaks_simu, peaks_salt, truth, match, peaks_salt_matched_to_simu, peaks_simu_matched_to_salt


def load_events(runs, st_salt, st_simu, plugins=('event_info', 'cuts_basic')):
    """
    Load events from the runs and do basic filtering suggeted by saltax.match_events
    :param runs: list of runs.
    :param st_salt: saltax context for salt mode
    :param st_simu: saltax context for simu mode
    :return: events_simu: events from simulated dataset, filtered out those who miss S1
    :return: events_salt: events from sprinkled dataset
    :return inds_dict: dictionary of indices of events from sprinkled or filtered simulated dataset, 
                       regarding matching events or s1 or s2
    """
    # Initialize the dictionary to store the indices
    inds_dict = {
        "ind_salt_event_found": np.array([], dtype=np.int32),
        "ind_salt_s1_found": np.array([], dtype=np.int32),
        "ind_salt_s2_found": np.array([], dtype=np.int32),
        "ind_simu_event_found": np.array([], dtype=np.int32),
        "ind_simu_s1_found": np.array([], dtype=np.int32),
        "ind_simu_s2_found": np.array([], dtype=np.int32)
    }

    for i, run in enumerate(runs):
        print("Loading run %s"%(run))
        
        # Load plugins for both salt and simu
        events_simu_i = st_simu.get_array(run, plugins, progress_bar=False)
        events_salt_i = st_salt.get_array(run, plugins, progress_bar=False)

        # Get matching result
        (
            events_simu_filtered_i,
            ind_salt_event_found_i, ind_simu_event_found_i, ind_simu_event_lost_i, ind_simu_event_split_i,
            ind_salt_s1_found_i, ind_simu_s1_found_i,
            ind_salt_s2_found_i, ind_simu_s2_found_i
        ) = saltax.match_events(events_simu_i, events_salt_i)

        # Load the indices into the dictionary
        inds_dict["ind_salt_event_found"] = np.concatenate(
            (inds_dict["ind_salt_event_found"], ind_salt_event_found_i)
        )
        inds_dict["ind_salt_s1_found"] = np.concatenate(
            (inds_dict["ind_salt_s1_found"], ind_salt_s1_found_i)
        )
        inds_dict["ind_salt_s2_found"] = np.concatenate(
            (inds_dict["ind_salt_s2_found"], ind_salt_s2_found_i)
        )
        inds_dict["ind_simu_event_found"] = np.concatenate(
            (inds_dict["ind_simu_event_found"], ind_simu_event_found_i)
        )
        inds_dict["ind_simu_s1_found"] = np.concatenate(
            (inds_dict["ind_simu_s1_found"], ind_simu_s1_found_i)
        )
        inds_dict["ind_simu_s2_found"] = np.concatenate(
            (inds_dict["ind_simu_s2_found"], ind_simu_s2_found_i)
        )

        # Concatenate the events
        if i==0:
            events_simu = events_simu_filtered_i
            events_salt = events_salt_i
        else:
            events_simu = np.concatenate((events_simu, events_simu_filtered_i))
            events_salt = np.concatenate((events_salt, events_salt_i))
    
    return events_simu, events_salt, inds_dict


def load_events_deprecated(runs, st_salt, st_simu, plugins=('event_info', 'cuts_basic')):
    """
    Load events from the runs and do basic filtering suggeted by saltax.match_events
    :param runs: list of runs.
    :param st_salt: saltax context for salt mode
    :param st_simu: saltax context for simu mode
    :return: events_simu: events from simulated dataset
    :return: events_salt: events from sprinkled dataset
    :return: truth: truth information in simulation
    :return: match: pema match information
    :return: events_salt_matched_to_simu: events related plugins from sprinkled matched to simulated
    :return: events_simu_matched_to_salt: events related plugins from simulated matched to sprinkled
    """
    for i, run in enumerate(runs):
        print("Loading run %s"%(run))
        
        events_simu_i = st_simu.get_array(run, plugins, progress_bar=False)
        events_salt_i = st_salt.get_array(run, plugins, progress_bar=False)
        truth_i = st_simu.get_array(run, 'truth', progress_bar=False)
        match_i = st_simu.get_array(run, 'match_acceptance_extended', progress_bar=False)

        # TODO: Ugly hardcoding for FV cut, need to be fixed
        events_salt_matched_to_simu_i, \
            events_simu_matched_to_salt_i = saltax.match_events(truth_i[(truth_i['z']<-13)&(truth_i['z']>-145)&(truth_i['x']**2+truth_i['y']**2<64**2)], 
                                                                match_i[(truth_i['z']<-13)&(truth_i['z']>-145)&(truth_i['x']**2+truth_i['y']**2<64**2)],
                                                                events_simu_i[(events_simu_i['z']<-13)&(events_simu_i['z']>-145)&(events_simu_i['r']<64)], 
                                                                events_salt_i[(events_salt_i['z']<-13)&(events_salt_i['z']>-145)])    
        if i==0:
            events_simu = events_simu_i
            events_salt = events_salt_i
            truth = truth_i
            match = match_i
            events_salt_matched_to_simu = events_salt_matched_to_simu_i
            events_simu_matched_to_salt = events_simu_matched_to_salt_i
        else:
            events_simu = np.concatenate((events_simu,events_simu_i))
            events_salt = np.concatenate((events_salt,events_salt_i))
            truth = np.concatenate((truth,truth_i))
            match = np.concatenate((match,match_i))
            events_salt_matched_to_simu = np.concatenate((events_salt_matched_to_simu,
                                                          events_salt_matched_to_simu_i))
            events_simu_matched_to_salt = np.concatenate((events_simu_matched_to_salt,
                                                          events_simu_matched_to_salt_i))
    
    return events_simu, events_salt, truth, match, events_salt_matched_to_simu, events_simu_matched_to_salt


def compare_templates(events_salt_matched_to_simu, events_simu_matched_to_salt, 
                      n_bins = 31, title='Ambience Interference in SR0 AmBe'):
    """
    Visually compare the cs1-cs2 templates of salted and simulated events.
    :param events_salt_matched_to_simu: events from saltax matched to simulation, with equal length
    :param events_simu_matched_to_salt: events from simulation matched to saltax, with equal length
    :param n_bins: number of bins for cs1
    :param title: title of the plot
    """
    cs1_bins = np.linspace(0,100,n_bins)
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
    
    for i in range(n_bins-1):
        selected_salt = events_salt_matched_to_simu[(events_salt_matched_to_simu['cs1']>=cs1_bins[i])*
                                                    (events_salt_matched_to_simu['cs1']<cs1_bins[i+1])]
        salt_med.append(np.median(selected_salt['cs2'][selected_salt['cs2']>0]))
        salt_1sig_l.append(np.percentile(selected_salt['cs2'][selected_salt['cs2']>0], 16.5))
        salt_1sig_u.append(np.percentile(selected_salt['cs2'][selected_salt['cs2']>0], 83.5))
        salt_2sig_l.append(np.percentile(selected_salt['cs2'][selected_salt['cs2']>0], 2.5))
        salt_2sig_u.append(np.percentile(selected_salt['cs2'][selected_salt['cs2']>0], 97.5))
        
        selected_simu = events_simu_matched_to_salt[(events_simu_matched_to_salt['cs1']>=cs1_bins[i])*
                                                    (events_simu_matched_to_salt['cs1']<cs1_bins[i+1])]
        simu_med.append(np.median(selected_simu['cs2'][selected_simu['cs2']>0]))
        simu_1sig_l.append(np.percentile(selected_simu['cs2'][selected_simu['cs2']>0], 16.5))
        simu_1sig_u.append(np.percentile(selected_simu['cs2'][selected_simu['cs2']>0], 83.5))
        simu_2sig_l.append(np.percentile(selected_simu['cs2'][selected_simu['cs2']>0], 2.5))
        simu_2sig_u.append(np.percentile(selected_simu['cs2'][selected_simu['cs2']>0], 97.5))
    
    cs1_coord = np.linspace(0,100,n_bins-1) + 100/2/(n_bins-1)
    
    plt.figure(dpi=150)
    plt.scatter(events_salt_matched_to_simu['cs1'],events_salt_matched_to_simu['cs2'],s=0.5, label='Sprinkled', alpha=0.2)
    plt.scatter(events_simu_matched_to_salt['cs1'],events_simu_matched_to_salt['cs2'],s=0.5, label='Simulated', alpha=0.2)
    plt.plot(cs1_coord, salt_med, color='tab:blue', label='Sprk Median')
    plt.plot(cs1_coord, simu_med, color='tab:red', label='Simu Median')
    plt.plot(cs1_coord, salt_1sig_l, color='tab:blue', label='Sprk 1sig',linestyle='dashed')
    plt.plot(cs1_coord, salt_1sig_u, color='tab:blue', linestyle='dashed')
    plt.plot(cs1_coord, salt_2sig_l, color='tab:blue', label='Sprk 2sig',linestyle='dashed', alpha=0.5)
    plt.plot(cs1_coord, salt_2sig_u, color='tab:blue', linestyle='dashed', alpha=0.5)
    plt.plot(cs1_coord, simu_1sig_l, color='tab:red', label='Simu 1sig',linestyle='dashed')
    plt.plot(cs1_coord, simu_1sig_u, color='tab:red', linestyle='dashed')
    plt.plot(cs1_coord, simu_2sig_l, color='tab:red', label='Simu 2sig',linestyle='dashed', alpha=0.5)
    plt.plot(cs1_coord, simu_2sig_u, color='tab:red', linestyle='dashed', alpha=0.5)
    
    
    plt.legend()
    plt.xlim(0,100)
    plt.ylim(0,6500)
    plt.xlabel('CS1 [PE]')
    plt.ylabel('CS2 [PE]')
    plt.title(title)
    plt.show()


def apply_n_minus_1_cuts(events_with_cuts, cut_oi, 
                         all_cuts = ALL_CUTS_EXCEPT_S2PatternS1Width):
    """
    Apply N-1 cuts to the events, where N is the number of cuts.
    :param events_with_cuts: events with cuts
    :param cut_oi: the cut to be left out for examination
    :param all_cuts: all cuts
    """
    other_cuts = all_cuts[all_cuts!=cut_oi]
    mask = np.ones(len(events_with_cuts), dtype=bool)
    
    for cut in other_cuts:
        mask &= events_with_cuts[cut]
    
    return mask


def apply_single_cut(events_with_cuts, cut_oi, all_cuts = None):
    """
    Apply a single cut to the events.
    :param events_with_cuts: events with cuts
    :param cut_oi: the cut to be applied
    :param all_cuts: pseudo parameter, not really used
    """
    mask = np.ones(len(events_with_cuts), dtype=bool)
    for cut in [cut_oi]:
        mask &= events_with_cuts[cut]
    return mask


def apply_cut_lists(events_with_cuts, 
                    all_cuts = ALL_CUTS_EXCEPT_S2PatternS1Width):
    """
    Apply a list of cuts to the events.
    :param events_with_cuts: events with cuts
    :param all_cuts: list of cuts to be applied
    """
    mask = np.ones(len(events_with_cuts), dtype=bool)
    for cut in all_cuts:
        mask &= events_with_cuts[cut]
    return mask


def get_n_minus_1_cut_acc(events_salt_matched_to_simu, events_simu_matched_to_salt,
                          all_cut_list = ALL_CUTS):
    """
    Get a text table of acceptance of N-1 cut acceptance for each cut.
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
        mask_salt_except_cut_oi = apply_n_minus_1_cuts(events_salt_matched_to_simu, cut_oi, 
                                                       all_cuts = all_cut_list)
        mask_simu_except_cut_oi = apply_n_minus_1_cuts(events_simu_matched_to_salt, cut_oi, 
                                                       all_cuts = all_cut_list)
        acceptance_salt = int(np.sum(mask_salt_all_cuts)/np.sum(mask_salt_except_cut_oi)*100)/100
        acceptance_simu = int(np.sum(mask_simu_all_cuts)/np.sum(mask_simu_except_cut_oi)*100)/100
        
        # Add a row for each cut
        table_data.append([cut_oi, acceptance_salt, acceptance_simu])
    
    # Define the headers
    headers = ['Cut Name', 'Acceptance in Sprk', 'Acceptance in Simu']
    
    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def get_single_cut_acc(events_salt_matched_to_simu, events_simu_matched_to_salt,
                       all_cut_list = ALL_CUTS):
    """
    Get a text table of acceptance of single cut acceptance for each cut.
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
        acceptance_salt = int(np.sum(mask_salt_except_cut_oi)/np.sum(mask_salt_no_cuts)*100)/100
        acceptance_simu = int(np.sum(mask_simu_except_cut_oi)/np.sum(mask_simu_no_cuts)*100)/100
        
        # Add a row for each cut
        table_data.append([cut_oi, acceptance_salt, acceptance_simu])
    
    # Define the headers
    headers = ['Cut Name', 'Acceptance in Sprk', 'Acceptance in Simu']
    
    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def get_cut_eff(events, all_cut_list = ALL_CUTS, 
                n_bins=31, coord = 'cs1', plot=True,
                indv_cut_type='n_minus_1', title='N-1 Cut Acceptance Measured in SR1 AmBe',
                bbox_to_anchor=(0.5, 1.50)):
    """
    Get the acceptance of each cut as a function of a coordinate.
    :param events: events
    :param all_cut_list: list of all cuts
    :param n_bins: number of bins for the coordinate
    :param coord: coordinate to be binned, default to 'cs1'
    :param plot: whether to plot the acceptance, default to True
    :param indv_cut_type: type of cut to be applied, default to 'n_minus_1', can also be 'single'
    :param title: title of the plot
    :param bbox_to_anchor: position of the legend, default to (0.5, 1.50)
    :return: a dictionary of acceptance values
    """
    coord_units = {'s1_area': '[PE]', 's2_area': '[PE]', 'cs1':'[PE]', 'cs2': '[PE]', 'z': '[cm]'}
    if coord == 'cs1' or coord == 's1_area':
        bins = np.linspace(0,100,n_bins)
    elif coord == 'cs2' or coord == 's2_area':
        bins = np.linspace(500,5000,n_bins)
    elif coord == 'z':
        bins = np.linspace(-145,-13,n_bins)
    else:
        raise NotImplementedError

    if indv_cut_type == 'n_minus_1':
        cut_func = apply_n_minus_1_cuts
    elif indv_cut_type == 'single':
        cut_func = apply_single_cut
    else:
        raise NotImplementedError

    result_dict = {}
    for cut in all_cut_list:
        result_dict[cut] = np.zeros(n_bins-1)
    result_dict['all_cuts'] = np.zeros(n_bins-1)
    result_dict[coord] = np.zeros(n_bins-1)
    for i in range(n_bins-1):
        mid_coord = (bins[i] + bins[i+1]) / 2
        result_dict[coord][i] = mid_coord
        
        selected_events = events[(events[coord]>=bins[i])*
                                 (events[coord]< bins[i+1])]
        selected_events_all_cut = selected_events[apply_cut_lists(selected_events, 
                                                                  all_cut_list)]
        result_dict['all_cuts'][i] = len(selected_events_all_cut)/len(selected_events)
        for cut_oi in all_cut_list:
            selected_events_cut_oi = selected_events[apply_single_cut(selected_events,
                                                                       cut_oi)]
            result_dict[cut_oi][i] = len(selected_events_cut_oi)/len(selected_events)
        
    if plot:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_cut_list) + 1))  # +1 for the 'Combined' line
        color_cycle = cycle(colors)
        plt.figure(dpi=150)
        plt.plot(result_dict[coord], result_dict['all_cuts'], color='k', label='Combined')
        for cut_oi in all_cut_list:
            plt.plot(result_dict[coord], result_dict[cut_oi], 
                     color=next(color_cycle), label=cut_oi)
        plt.xlabel(coord+coord_units[coord])
        plt.ylabel('Measured Acceptance')
        plt.grid()
        plt.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, 
                   ncol=2, fontsize='small', frameon=False)
        plt.title(title)

    return result_dict


def compare_2d(events0, events1, bins, title, xlim, ylim, 
               label0, label1, xlabel, ylabel, 
               coords=['z', 's2_range_50p_area']):
    """
    Compare 2D distributions of two datasets.
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
    
    for i in range(len(bins)-1):
        mask0_i = (events0[coords[0]]>=bins[i])&(events0[coords[0]]<bins[i+1])
        events0_i = events0[mask0_i]
        events0_med.append(np.median(events0_i[coords[1]]))
        events0_1sig_l.append(np.percentile(events0_i[coords[1]], 16.5))
        events0_1sig_u.append(np.percentile(events0_i[coords[1]], 83.5))
        events0_2sig_l.append(np.percentile(events0_i[coords[1]], 2.5))
        events0_2sig_u.append(np.percentile(events0_i[coords[1]], 97.5))

        mask1_i = (events1[coords[0]]>=bins[i])&(events1[coords[0]]<bins[i+1])
        events1_i = events1[mask1_i]
        events1_med.append(np.median(events1_i[coords[1]]))
        events1_1sig_l.append(np.percentile(events1_i[coords[1]], 16.5))
        events1_1sig_u.append(np.percentile(events1_i[coords[1]], 83.5))
        events1_2sig_l.append(np.percentile(events1_i[coords[1]], 2.5))
        events1_2sig_u.append(np.percentile(events1_i[coords[1]], 97.5))
    
    bins_mid = (bins[:-1]+bins[1:])/2
    
    plt.figure(dpi=150)
    
    plt.scatter(events0[coords[0]], events0[coords[1]], s=0.5, alpha=0.2)
    plt.scatter(events1[coords[0]], events1[coords[1]], s=0.5, alpha=0.2)
    
    plt.plot(bins_mid, events0_med, color='tab:blue', label=label0)
    plt.plot(bins_mid, events1_med, color='tab:red', label=label1)
    plt.plot(bins_mid, events0_1sig_l, color='tab:blue', linestyle='dashed')
    plt.plot(bins_mid, events0_1sig_u, color='tab:blue', linestyle='dashed')
    plt.plot(bins_mid, events0_2sig_l, color='tab:blue', linestyle='dashed', alpha=0.5)
    plt.plot(bins_mid, events0_2sig_u, color='tab:blue', linestyle='dashed', alpha=0.5)
    plt.plot(bins_mid, events1_1sig_l, color='tab:red', linestyle='dashed')
    plt.plot(bins_mid, events1_1sig_u, color='tab:red', linestyle='dashed')
    plt.plot(bins_mid, events1_2sig_l, color='tab:red',linestyle='dashed', alpha=0.5)
    plt.plot(bins_mid, events1_2sig_u, color='tab:red', linestyle='dashed', alpha=0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.show()


def compare_bands(salt, simu, title,
                  coords=['z', 's2_range_50p_area'], n_bins=16):
    """
    Wrapped comparison of 2D distributions of two datasets.
    :param salt: events from the first dataset
    :param simu: events from the second dataset
    :param title: title of the plot
    :param coords: coordinates to be compared, default to ['z', 's2_range_50p_area'], can choose from ['z', 's1_area', 's2_area', 's1_range_50p_area', 's1_range_90p_area', 's1_rise_time', 's2_range_50p_area', 's2_range_90p_area']
    :param n_bins: number of bins for each coordinate, default to 16
    """
    BINS = {
    'z': np.linspace(-134,-13,n_bins),
    's1_area': np.linspace(0,100,n_bins),
    's2_area': np.linspace(500,5000,n_bins),
    's1_range_50p_area': np.linspace(0,300,n_bins),
    's1_range_90p_area': np.linspace(0,1000,n_bins),
    's1_rise_time': np.linspace(0,150,n_bins),
    's2_range_50p_area': np.linspace(0,1.5e4,n_bins),
    's2_range_90p_area': np.linspace(0,4e4,n_bins),
    }
    UNITS_DICT = {
        'z': '[cm]',
        's1_area': '[PE]',
        's2_area': '[PE]',
        's1_range_50p_area': '[ns]',
        's1_range_90p_area': '[ns]',
        's1_rise_time': '[ns]',
        's2_range_50p_area': '[ns]',
        's2_range_90p_area': '[ns]',
    }
    bins = BINS
    units_dict = UNITS_DICT
    
    compare_2d(salt, simu, 
               bins=bins[coords[0]], 
               title=title, 
               xlim=(bins[coords[0]][0],bins[coords[0]][-1]), 
               ylim=(bins[coords[1]][0],bins[coords[1]][-1]), 
               label0='Sprinkled', 
               label1='Simulated',
               xlabel=coords[0]+units_dict[coords[0]],
               ylabel=coords[1]+units_dict[coords[1]],
               coords=coords)


def show_area_bias(salt, simu, title, 
                   coord='s1_area', s1s2='S1', n_bins=16, ylim=(-5,20)):
    """
    Show the bias due to ambience interference VS a coordinate.
    :param salt: events from the first dataset
    :param simu: events from the second dataset
    :param title: title of the plot
    :param coord: coordinate to be compared, default to 's1_area', can choose from ['z', 's1_area', 's2_area', 's1_range_50p_area', 's1_range_90p_area', 's1_rise_time', 's2_range_50p_area', 's2_range_90p_area']
    :param s1s2: S1 or S2, default to 'S1'
    :param n_bins: number of bins for the coordinate, default to 16
    :param ylim: y-axis limits, default to (-5,20)
    """
    BINS = {
    'z': np.linspace(-134,-13,n_bins),
    's1_area': np.linspace(0,100,n_bins),
    's2_area': np.linspace(500,5000,n_bins),
    's1_range_50p_area': np.linspace(0,300,n_bins),
    's1_range_90p_area': np.linspace(0,1000,n_bins),
    's1_rise_time': np.linspace(0,150,n_bins),
    's2_range_50p_area': np.linspace(0,1.5e4,n_bins),
    's2_range_90p_area': np.linspace(0,4e4,n_bins),
    }
    UNITS_DICT = {
        'z': '[cm]',
        's1_area': '[PE]',
        's2_area': '[PE]',
        's1_range_50p_area': '[ns]',
        's1_range_90p_area': '[ns]',
        's1_rise_time': '[ns]',
        's2_range_50p_area': '[ns]',
        's2_range_90p_area': '[ns]',
    }
    bins = BINS[coord]
    units_dict = UNITS_DICT
    if s1s2 == 'S1':
        bias = salt['s1_area'] - simu['s1_area']
    elif s1s2 == 'S2':
        bias = salt['s2_area'] - simu['s2_area']
    else:
        raise ValueError

    bins_mid = (bins[1:]+bins[:-1])/2
    bias_med = []
    bias_1sig_u = []
    bias_1sig_l = []
    bias_2sig_u = []
    bias_2sig_l = []

    for i in range(n_bins-1):
        mask0_i = (simu[coord]>=bins[i])&(simu[coord]<bins[i+1])
        bias_i = bias[mask0_i]
        bias_med.append(np.median(bias_i))
        bias_1sig_l.append(np.percentile(bias_i, 16.5))
        bias_1sig_u.append(np.percentile(bias_i, 83.5))
        bias_2sig_l.append(np.percentile(bias_i, 2.5))
        bias_2sig_u.append(np.percentile(bias_i, 97.5))

    plt.figure(dpi=150)
    plt.scatter(simu[coord], bias, s=0.5, alpha=0.2, color='k')
    plt.plot(bins_mid, bias_med, color='tab:blue', label='Median')
    plt.plot(bins_mid, bias_1sig_l, color='tab:blue', linestyle='dashed', label='1Sig')
    plt.plot(bins_mid, bias_1sig_u, color='tab:blue', linestyle='dashed')
    plt.plot(bins_mid, bias_2sig_l, color='tab:blue', linestyle='dashed', alpha=0.5, label='2Sig')
    plt.plot(bins_mid, bias_2sig_u, color='tab:blue', linestyle='dashed', alpha=0.5)

    plt.xlabel(coord+units_dict[coord])
    plt.ylabel('Change in %s Area [PE]'%(s1s2))
    plt.xlim(bins[0], bins[-1])
    plt.legend()
    plt.ylim(ylim)
    plt.title(title)
    plt.show()
