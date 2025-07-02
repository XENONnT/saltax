from straxen.test_utils import nt_test_run_id

from saltax.match.utils import *
from saltax.match.visual import plot_event_wf
from saltax.test_utils import TEST_DATA_TYPES, get_test_context


def test_match():
    """Test match module."""

    # Init contexts for both salt and simu modes
    st = {}
    for saltax_mode in ["salt", "simu", "data"]:
        st[saltax_mode] = get_test_context(saltax_mode)

    # Try creating some data_types in both modes
    for dt in TEST_DATA_TYPES:
        for saltax_mode in ["salt", "simu", "data"]:
            st[saltax_mode].make(nt_test_run_id, dt, save=dt)

    runs_with_rawdata = find_runs_with_rawdata(rawdata_folders=[st["salt"].storage[-1].path])

    get_available_runs(
        runs_with_rawdata,
        st["salt"],
        st["simu"],
        salt_available=["event_basics"],
        simu_available=["event_basics"],
    )

    (peaks_simu, peaks_salt, inds_dict) = load_peaks([nt_test_run_id], st["salt"], st["simu"])
    peaks_simu_matched_to_salt = peaks_simu[inds_dict["ind_simu_peak_found"]]

    (events_simu, events_salt, inds_dict) = load_events([nt_test_run_id], st["salt"], st["simu"])
    events_salt_matched_to_simu = events_salt[inds_dict["ind_salt_s1_found"]]
    events_simu_matched_to_salt = events_simu[inds_dict["ind_simu_s1_found"]]

    cut_list = ["cut_daq_veto", "cut_interaction_exists"]

    mask_salt_cut = apply_cut_lists(events_salt_matched_to_simu, cut_list)
    mask_simu_cut = apply_cut_lists(events_simu_matched_to_salt, cut_list)

    n_bins = 2

    compare_templates(
        events_salt_matched_to_simu[mask_salt_cut],
        events_simu_matched_to_salt[mask_simu_cut],
        n_bins=n_bins,
    )

    get_n_minus_1_cut_acc(
        events_salt_matched_to_simu, events_simu_matched_to_salt, all_cut_list=cut_list
    )

    get_single_cut_acc(
        events_salt_matched_to_simu, events_simu_matched_to_salt, all_cut_list=cut_list
    )

    get_cut_eff(events_salt_matched_to_simu, all_cut_list=cut_list, n_bins=n_bins)

    compare_bands(
        events_salt_matched_to_simu[mask_salt_cut],
        events_simu_matched_to_salt[mask_salt_cut],
        title="Title",
        n_bins=n_bins,
    )

    show_area_bias(
        events_salt_matched_to_simu[mask_salt_cut & mask_simu_cut],
        events_simu_matched_to_salt[mask_salt_cut & mask_simu_cut],
        title="Title",
        n_bins=n_bins,
    )

    show_eff1d(events_simu, events_simu_matched_to_salt, bins=np.linspace(0, 12, n_bins))
    show_eff2d(
        events_simu,
        events_simu_matched_to_salt,
        bins=(np.linspace(0, 100, n_bins), np.linspace(500, 7000, n_bins)),
    )

    apply_peaks_daq_cuts(st["data"], [nt_test_run_id], peaks_simu_matched_to_salt)

    plot_event_wf(
        ind=0,
        st_salt=st["salt"],
        st_simu=st["simu"],
        st_data=st["data"],
        runid=nt_test_run_id,
        events_simu=events_simu_matched_to_salt[mask_simu_cut],
        events_salt=events_salt_matched_to_simu[mask_salt_cut],
    )
