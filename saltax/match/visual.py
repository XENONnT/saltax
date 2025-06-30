import numpy as np
import matplotlib.pyplot as plt


def plot_event_wf(
    ind,
    st_salt,
    st_simu,
    st_data,
    runid,
    events_simu,
    events_salt=None,
    event_ext_window_ns=2.4e6,
    s1_ext_window_samples=25,
    s2_ext_window_samples=100,
    ylim=(0, 5),
):
    """Plot waveforms for a single event, including full event waveform and zoomed-in S1 and S2
    waveforms.

    This plot function will show info from salt, simu and data mode.
    :param ind: index of the event in the events_simu dataframe
    :param st_salt: saltax context for salt mode
    :param st_simu: saltax context for simu mode
    :param st_data: saltax context for data mode
    :param runid: runid of the event, example: '066666'
    :param events_simu: simu event_info.
    :param events_salt: salt event_info matched to events_simu, default None and event level
        information from sprinkled dataset won't be used
    :param event_ext_window_ns: time window in ns to plot around the event, default 2.4e6 ns = 2.4
        ms
    :param s1_ext_window_samples: time window in samples to plot around S1, default 25 samples
    :param s2_ext_window_samples: time window in samples to plot around S2, default 100 samples
    :param ylim: y-axis limits for the waveforms, default (0,5) PE/10ns

    """
    if events_salt is not None:
        assert len(events_salt) == len(
            events_simu
        ), "events_salt and events_simu should have the same length, \
            since they are expected to 1-1 matched"

    print("Loading peaks and lone_hits for run %s event %s" % (runid, ind))

    # Get time ranges in indices for events, S1 and S2
    extended_simu_event_timerange_ns = (
        events_simu["s1_time"][ind] - event_ext_window_ns,
        events_simu["s2_endtime"][ind] + event_ext_window_ns,
    )
    matched_simu_s1_timerange_i = (
        int((events_simu["s1_time"][ind] - extended_simu_event_timerange_ns[0]) / 10),
        int((events_simu["s1_endtime"][ind] - extended_simu_event_timerange_ns[0]) / 10),
    )
    matched_simu_s2_timerange_i = (
        int((events_simu["s2_time"][ind] - extended_simu_event_timerange_ns[0]) / 10),
        int((events_simu["s2_endtime"][ind] - extended_simu_event_timerange_ns[0]) / 10),
    )
    if events_salt is not None:
        matched_salt_s1_timerange_i = (
            int((events_salt["s1_time"][ind] - extended_simu_event_timerange_ns[0]) / 10),
            int((events_salt["s1_endtime"][ind] - extended_simu_event_timerange_ns[0]) / 10),
        )
        matched_salt_s2_timerange_i = (
            int((events_salt["s2_time"][ind] - extended_simu_event_timerange_ns[0]) / 10),
            int((events_salt["s2_endtime"][ind] - extended_simu_event_timerange_ns[0]) / 10),
        )
    simu_event_timerange_i = (
        int((events_simu["time"][ind] - extended_simu_event_timerange_ns[0]) / 10),
        int((events_simu["endtime"][ind] - extended_simu_event_timerange_ns[0]) / 10),
    )
    salt_event_timerange_i = (
        int((events_salt["time"][ind] - extended_simu_event_timerange_ns[0]) / 10),
        int((events_salt["endtime"][ind] - extended_simu_event_timerange_ns[0]) / 10),
    )

    # Get peaks and lone hits for the event
    # Make sure the data is stored before loading
    context_dict = {"salt": st_salt, "simu": st_simu, "data": st_data}
    for context_mode in context_dict.keys():
        st = context_dict[context_mode]
        for target in ["lone_hits", "peaklets", "peaklet_classification", "merged_s2s"]:
            assert st.is_stored(runid, target), "Data not stored for %s in %s mode" % (
                st.key_for(runid, target),
                str(context_mode),
            )
    # Actual data loading
    peaks_salt_selected = st_salt.get_array(
        runid, "peaks", time_range=extended_simu_event_timerange_ns, progress_bar=False
    )
    peaks_simu_selected = st_simu.get_array(
        runid, "peaks", time_range=extended_simu_event_timerange_ns, progress_bar=False
    )
    peaks_data_selected = st_data.get_array(
        runid, "peaks", time_range=extended_simu_event_timerange_ns, progress_bar=False
    )
    lhs_salt_selected = st_salt.get_array(
        runid, "lone_hits", time_range=extended_simu_event_timerange_ns, progress_bar=False
    )
    lhs_simu_selected = st_simu.get_array(
        runid, "lone_hits", time_range=extended_simu_event_timerange_ns, progress_bar=False
    )
    lhs_data_selected = st_data.get_array(
        runid, "lone_hits", time_range=extended_simu_event_timerange_ns, progress_bar=False
    )

    # Get waveforms for the event
    print("Building waveforms...")
    total_length = int(
        (extended_simu_event_timerange_ns[1] - extended_simu_event_timerange_ns[0]) / 10
    )
    to_pes = st_data.get_single_plugin(runid, "peaklets").to_pe
    # Initialize waveforms
    wf_salt_s1 = np.zeros(total_length)
    wf_simu_s1 = np.zeros(total_length)
    wf_salt_s2 = np.zeros(total_length)
    wf_simu_s2 = np.zeros(total_length)
    wf_salt_others = np.zeros(total_length)
    wf_simu_others = np.zeros(total_length)
    wf_data = np.zeros(total_length)
    # Fill sprinkled waveforms with peaks and lone hits
    if len(peaks_salt_selected):
        for p in peaks_salt_selected:
            start_i = int((p["time"] - int(extended_simu_event_timerange_ns[0])) / 10)
            length = p["length"]
            dt = p["dt"]
            if p["type"] == 1:
                for i in range(length):
                    wf_salt_s1[start_i + i * int(dt / 10) : start_i + (i + 1) * int(dt / 10)] = (
                        p["data"][i] / dt * 10
                    )
            elif p["type"] == 2:
                for i in range(length):
                    wf_salt_s2[start_i + i * int(dt / 10) : start_i + (i + 1) * int(dt / 10)] = (
                        p["data"][i] / dt * 10
                    )
            else:
                for i in range(length):
                    wf_salt_others[
                        start_i + i * int(dt / 10) : start_i + (i + 1) * int(dt / 10)
                    ] = (p["data"][i] / dt * 10)
    if len(lhs_salt_selected):
        for lh in lhs_salt_selected:
            time_i = int((lh["time"] - int(extended_simu_event_timerange_ns[0])) / 10)
            amp = lh["area"] * to_pes[lh["channel"]]
            wf_salt_others[time_i] += amp / 10
    # Fill simulated waveforms with peaks and lone hits
    if len(peaks_simu_selected):
        for p in peaks_simu_selected:
            start_i = int((p["time"] - int(extended_simu_event_timerange_ns[0])) / 10)
            length = p["length"]
            dt = p["dt"]
            if p["type"] == 1:
                for i in range(length):
                    wf_simu_s1[start_i + i * int(dt / 10) : start_i + (i + 1) * int(dt / 10)] = (
                        p["data"][i] / dt * 10
                    )
            elif p["type"] == 2:
                for i in range(length):
                    wf_simu_s2[start_i + i * int(dt / 10) : start_i + (i + 1) * int(dt / 10)] = (
                        p["data"][i] / dt * 10
                    )
            else:
                for i in range(length):
                    wf_simu_others[
                        start_i + i * int(dt / 10) : start_i + (i + 1) * int(dt / 10)
                    ] = (p["data"][i] / dt * 10)
    if len(lhs_simu_selected):
        for lh in lhs_simu_selected:
            time_i = int((lh["time"] - int(extended_simu_event_timerange_ns[0])) / 10)
            amp = lh["area"] * to_pes[lh["channel"]]
            wf_simu_others[time_i] += amp / 10
    # Fill data waveform with peaks and lone hits
    if len(peaks_data_selected):
        for p in peaks_data_selected:
            start_i = int((p["time"] - int(extended_simu_event_timerange_ns[0])) / 10)
            length = p["length"]
            dt = p["dt"]
            for i in range(length):
                wf_data[start_i + i * int(dt / 10) : start_i + (i + 1) * int(dt / 10)] = (
                    p["data"][i] / dt * 10
                )
    if len(lhs_data_selected):
        for lh in lhs_data_selected:
            time_i = int((lh["time"] - int(extended_simu_event_timerange_ns[0])) / 10)
            amp = lh["area"] * to_pes[lh["channel"]]
            wf_data[time_i] += amp / 10

    # Plot full event waveform
    print("Plotting waveforms for the whole event...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), dpi=150)
    # Plot waveforms for simulated dataset
    ax1.plot(wf_data, label="Data", color="k", alpha=0.5)
    ax1.plot(wf_simu_s1, label="Simulated S1", color="tab:blue")
    ax1.plot(wf_simu_s2, label="Simulated S2", color="tab:red")
    ax1.plot(wf_simu_others, label="Simulated Others", color="tab:green")
    ax1.axvspan(
        matched_simu_s1_timerange_i[0],
        matched_simu_s1_timerange_i[1],
        color="tab:blue",
        alpha=0.2,
        label="Simulated S1 Range",
    )
    ax1.axvspan(
        matched_simu_s2_timerange_i[0],
        matched_simu_s2_timerange_i[1],
        color="tab:red",
        alpha=0.2,
        label="Simulated S2 Range",
    )
    ax1.axvspan(
        simu_event_timerange_i[0],
        simu_event_timerange_i[1],
        color="tab:brown",
        alpha=0.2,
        label="Simulated Event",
    )
    if events_salt is not None:
        ax1.axvspan(
            matched_salt_s1_timerange_i[0],
            matched_salt_s1_timerange_i[1],
            color="b",
            alpha=0.2,
            label="Sprinkled S1 Range",
        )
        ax1.axvspan(
            matched_salt_s2_timerange_i[0],
            matched_salt_s2_timerange_i[1],
            color="r",
            alpha=0.2,
            label="Sprinkled S2 Range",
        )
        ax1.axvspan(
            salt_event_timerange_i[0],
            salt_event_timerange_i[1],
            color="grey",
            alpha=0.1,
            label="Matched Sprinkled Event",
        )
    ax1.legend()
    if events_salt is not None:
        ax1.set_title(
            "Run %s Event %s: Simu/Sprk S1=%s/%sPE, Simu/Sprk S2=%s/%sPE"
            % (
                runid,
                ind,
                int(10 * events_simu["s1_area"][ind]) / 10,
                int(10 * events_salt["s1_area"][ind]) / 10,
                int(10 * events_simu["s2_area"][ind]) / 10,
                int(10 * events_salt["s2_area"][ind]) / 10,
            )
        )
    else:
        ax1.set_title(
            "Run %s Event %s: Simu S1=%sPE, S2=%sPE"
            % (
                runid,
                ind,
                int(10 * events_simu["s1_area"][ind]) / 10,
                int(10 * events_simu["s2_area"][ind]) / 10,
            )
        )
    ax1.set_ylim(ylim)
    # Plot waveforms for sprinkeld dataset
    ax2.plot(wf_data, label="Data", color="k", alpha=0.5)
    ax2.plot(wf_salt_s1, label="Sprinkled S1", color="b")
    ax2.plot(wf_salt_s2, label="Sprinkled S2", color="r")
    ax2.plot(wf_salt_others, label="Sprinkled Others", color="g")
    ax2.axvspan(
        matched_simu_s1_timerange_i[0],
        matched_simu_s1_timerange_i[1],
        color="tab:blue",
        alpha=0.2,
        label="Simulated S1 Range",
    )
    ax2.axvspan(
        matched_simu_s2_timerange_i[0],
        matched_simu_s2_timerange_i[1],
        color="tab:red",
        alpha=0.2,
        label="Simulated S2 Range",
    )
    ax2.axvspan(
        simu_event_timerange_i[0],
        simu_event_timerange_i[1],
        color="tab:brown",
        alpha=0.2,
        label="Simulated Event",
    )
    if events_salt is not None:
        ax2.axvspan(
            matched_salt_s1_timerange_i[0],
            matched_salt_s1_timerange_i[1],
            color="b",
            alpha=0.2,
            label="Sprinkled S1 Range",
        )
        ax2.axvspan(
            matched_salt_s2_timerange_i[0],
            matched_salt_s2_timerange_i[1],
            color="r",
            alpha=0.2,
            label="Sprinkled S2 Range",
        )
        ax2.axvspan(
            salt_event_timerange_i[0],
            salt_event_timerange_i[1],
            color="grey",
            alpha=0.1,
            label="Matched Sprinkled Event",
        )
    ax2.legend()
    ax2.set_ylim(ylim)
    # Set labels for full event waveform plot
    ax1.set_ylabel("Amplitude [PE/10ns]")
    ax2.set_xlabel("Time [10ns]")
    ax2.set_ylabel("Amplitude [PE/10ns]")
    fig.show()

    # Zoom into S1 and S2 waveforms
    print("Zooming into S1 and S2 respectively...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), dpi=150)
    # Plot zoomed-in waveforms for simulated dataset around main S1
    axs[0, 0].plot(wf_data, label="Data", color="k", alpha=0.5)
    axs[0, 0].plot(wf_simu_s1, label="Simulated S1", color="tab:blue")
    axs[0, 0].plot(wf_simu_s2, label="Simulated S2", color="tab:red")
    axs[0, 0].plot(wf_simu_others, label="Simulated Others", color="tab:green")
    axs[0, 0].axvspan(
        matched_simu_s1_timerange_i[0],
        matched_simu_s1_timerange_i[1],
        color="tab:blue",
        alpha=0.2,
        label="Simulated S1 Range",
    )
    axs[0, 0].axvspan(
        matched_simu_s2_timerange_i[0],
        matched_simu_s2_timerange_i[1],
        color="tab:red",
        alpha=0.2,
        label="Simulated S2 Range",
    )
    if events_salt is not None:
        axs[0, 0].axvspan(
            matched_salt_s1_timerange_i[0],
            matched_salt_s1_timerange_i[1],
            color="b",
            alpha=0.2,
            label="Sprinkled S1 Range",
        )
        axs[0, 0].axvspan(
            matched_salt_s2_timerange_i[0],
            matched_salt_s2_timerange_i[1],
            color="r",
            alpha=0.2,
            label="Sprinkled S2 Range",
        )
    axs[0, 0].set_xlim(
        matched_simu_s1_timerange_i[0] - s1_ext_window_samples,
        matched_simu_s1_timerange_i[1] + s1_ext_window_samples,
    )
    axs[0, 0].set_ylabel("Amplitude [PE/10ns]")
    axs[0, 0].legend()
    axs[0, 0].set_ylim(ylim)
    # Plot zoomed-in waveforms for simulated dataset around main S2
    axs[0, 1].plot(wf_data, label="Data", color="k", alpha=0.5)
    axs[0, 1].plot(wf_simu_s1, label="Simulated S1", color="tab:blue")
    axs[0, 1].plot(wf_simu_s2, label="Simulated S2", color="tab:red")
    axs[0, 1].plot(wf_simu_others, label="Simulated Others", color="tab:green")
    axs[0, 1].axvspan(
        matched_simu_s1_timerange_i[0],
        matched_simu_s1_timerange_i[1],
        color="tab:blue",
        alpha=0.2,
        label="Simulated S1 Range",
    )
    axs[0, 1].axvspan(
        matched_simu_s2_timerange_i[0],
        matched_simu_s2_timerange_i[1],
        color="tab:red",
        alpha=0.2,
        label="Simulated S2 Range",
    )
    if events_salt is not None:
        axs[0, 1].axvspan(
            matched_salt_s1_timerange_i[0],
            matched_salt_s1_timerange_i[1],
            color="b",
            alpha=0.2,
            label="Sprinkled S1 Range",
        )
        axs[0, 1].axvspan(
            matched_salt_s2_timerange_i[0],
            matched_salt_s2_timerange_i[1],
            color="r",
            alpha=0.2,
            label="Sprinkled S2 Range",
        )
    axs[0, 1].set_xlim(
        matched_simu_s2_timerange_i[0] - s2_ext_window_samples,
        matched_simu_s2_timerange_i[1] + s2_ext_window_samples,
    )
    axs[0, 1].legend()
    axs[0, 1].set_ylim(ylim)
    # Plot zoomed-in waveforms for sprinkled dataset around main S1
    axs[1, 0].plot(wf_data, label="Data", color="k", alpha=0.5)
    axs[1, 0].plot(wf_salt_s1, label="Sprinkled S1", color="b")
    axs[1, 0].plot(wf_salt_s2, label="Sprinkled S2", color="r")
    axs[1, 0].plot(wf_salt_others, label="Sprinkled Others", color="g")
    axs[1, 0].axvspan(
        matched_simu_s1_timerange_i[0],
        matched_simu_s1_timerange_i[1],
        color="tab:blue",
        alpha=0.2,
        label="Simulated S1 Range",
    )
    axs[1, 0].axvspan(
        matched_simu_s2_timerange_i[0],
        matched_simu_s2_timerange_i[1],
        color="tab:red",
        alpha=0.2,
        label="Simulated S2 Range",
    )
    axs[1, 0].set_xlim(
        matched_simu_s1_timerange_i[0] - s1_ext_window_samples,
        matched_simu_s1_timerange_i[1] + s1_ext_window_samples,
    )
    if events_salt is not None:
        axs[1, 0].axvspan(
            matched_salt_s1_timerange_i[0],
            matched_salt_s1_timerange_i[1],
            color="b",
            alpha=0.2,
            label="Sprinkled S1 Range",
        )
        axs[1, 0].axvspan(
            matched_salt_s2_timerange_i[0],
            matched_salt_s2_timerange_i[1],
            color="r",
            alpha=0.2,
            label="Sprinkled S2 Range",
        )
        axs[1, 0].set_xlim(
            matched_salt_s1_timerange_i[0] - s1_ext_window_samples,
            matched_salt_s1_timerange_i[1] + s1_ext_window_samples,
        )
    axs[1, 0].set_xlabel("Time [10ns]")
    axs[1, 0].set_ylabel("Amplitude [PE/10ns]")
    axs[1, 0].legend()
    axs[1, 0].set_ylim(ylim)
    # Plot zoomed-in waveforms for sprinkled dataset around main S2
    axs[1, 1].plot(wf_data, label="Data", color="k", alpha=0.5)
    axs[1, 1].plot(wf_salt_s1, label="Sprinkled S1", color="b")
    axs[1, 1].plot(wf_salt_s2, label="Sprinkled S2", color="r")
    axs[1, 1].plot(wf_salt_others, label="Sprinkled Others", color="g")
    axs[1, 1].axvspan(
        matched_simu_s1_timerange_i[0],
        matched_simu_s1_timerange_i[1],
        color="tab:blue",
        alpha=0.2,
        label="Simulated S1 Range",
    )
    axs[1, 1].axvspan(
        matched_simu_s2_timerange_i[0],
        matched_simu_s2_timerange_i[1],
        color="tab:red",
        alpha=0.2,
        label="Simulated S2 Range",
    )
    axs[1, 1].set_xlim(
        matched_simu_s2_timerange_i[0] - s2_ext_window_samples,
        matched_simu_s2_timerange_i[1] + s2_ext_window_samples,
    )
    if events_salt is not None:
        axs[1, 1].axvspan(
            matched_salt_s1_timerange_i[0],
            matched_salt_s1_timerange_i[1],
            color="b",
            alpha=0.2,
            label="Sprinkled S1 Range",
        )
        axs[1, 1].axvspan(
            matched_salt_s2_timerange_i[0],
            matched_salt_s2_timerange_i[1],
            color="r",
            alpha=0.2,
            label="Sprinkled S2 Range",
        )
        axs[1, 1].set_xlim(
            matched_salt_s2_timerange_i[0] - s2_ext_window_samples,
            matched_salt_s2_timerange_i[1] + s2_ext_window_samples,
        )
    axs[1, 1].set_xlabel("Time [10ns]")
    axs[1, 1].legend()
    axs[1, 1].set_ylim(ylim)
    fig.show()
