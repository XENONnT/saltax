import straxen
import saltax
import cutax
import strax
from immutabledict import immutabledict
import fuse
import logging

# import pema
import pandas as pd

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.context")


# fuse plugins
# Plugins to simulate microphysics
MICROPHYSICS_PLUGINS = fuse.context.microphysics_plugins
# Plugins to simulate S1 signals
S1_SIMULATION_PLUGINS = fuse.context.s1_simulation_plugins
# Plugins to simulate S2 signals
S2_SIMULATION_PLUGINS = fuse.context.s2_simulation_plugins
# Plugins to simulate delayed electrons
DELAYED_ELECTRON_SIMULATION_PLUGINS = fuse.context.delayed_electron_simulation_plugins
# Plugins to merge delayed and regular electrons
DELAYED_ELECTRON_MERGER_PLUGINS = fuse.context.delayed_electron_merger_plugins
# Plugins to simulate PMTs and DAQ
# TODO: this plugin fuse.pmt_and_daq.PMTResponseAndDAQ will be replaced
PMT_AND_DAQ_PLUGINS = fuse.context.pmt_and_daq_plugins
# Plugins to get truth information
TRUTH_INFORMATION_PLUGINS = fuse.context.truth_information_plugins
# All plugins with fuse
FUSED_PLUGINS = [
    MICROPHYSICS_PLUGINS,
    S1_SIMULATION_PLUGINS,
    S2_SIMULATION_PLUGINS,
    DELAYED_ELECTRON_SIMULATION_PLUGINS,
    DELAYED_ELECTRON_MERGER_PLUGINS,
    PMT_AND_DAQ_PLUGINS,
    TRUTH_INFORMATION_PLUGINS,
]

# ~Infinite raw_records file size to avoid downchunking
MAX_RAW_RECORDS_FILE_SIZE_MB = 1e9

# fuse placeholder parameters
CORRECTION_RUN_ID_DEFAULT = "046477"

# straxen XENONnT options/configuration
XNT_COMMON_OPTS = straxen.contexts.xnt_common_opts.copy()
XNT_COMMON_CONFIG = straxen.contexts.xnt_common_config.copy()
XNT_SIMULATION_CONFIG = straxen.contexts.xnt_simulation_config.copy()

# saltax options overrides
SXNT_COMMON_OPTS_REGISTER = XNT_COMMON_OPTS["register"].copy()
SXNT_COMMON_OPTS_REGISTER.remove(straxen.PulseProcessing)
SXNT_COMMON_OPTS_REGISTER = [
    saltax.SPulseProcessing,
    saltax.SRawRecordsFromFaxNT,
] + SXNT_COMMON_OPTS_REGISTER
SXNT_COMMON_OPTS_OVERRIDE = dict(
    register=SXNT_COMMON_OPTS_REGISTER,
)
SXNT_COMMON_OPTS = XNT_COMMON_OPTS.copy()
SXNT_COMMON_OPTS["register"] = SXNT_COMMON_OPTS_OVERRIDE["register"]


# saltax configuration overrides
SCHANNEL_STARTS_AT = 3000
XNT_COMMON_CONFIG_OVERRIDE = dict(
    channel_map=immutabledict(
        # (Minimum channel, maximum channel)
        # Channels must be listed in a ascending order!
        stpc=(SCHANNEL_STARTS_AT, SCHANNEL_STARTS_AT + 493),  # Salted TPC channels
        tpc=(0, 493),  # TPC channels
        he=(500, 752),  # high energy
        aqmon=(790, 807),
        aqmon_nv=(808, 815),  # nveto acquisition monitor
        tpc_blank=(999, 999),
        mv=(1000, 1083),
        aux_mv=(1084, 1087),  # Aux mv channel 2 empty  1 pulser  and 1 GPS
        mv_blank=(1999, 1999),
        nveto=(2000, 2119),
        nveto_blank=(2999, 2999),
    ),
)
SXNT_COMMON_CONFIG = XNT_COMMON_CONFIG.copy()
SXNT_COMMON_CONFIG["channel_map"] = XNT_COMMON_CONFIG_OVERRIDE["channel_map"]
FXNT_COMMON_CONFIG = SXNT_COMMON_CONFIG
DEFAULT_XEDOCS_VERSION = cutax.contexts.DEFAULT_XEDOCS_VERSION

# saltax modes supported
SALTAX_MODES = ["data", "simu", "salt"]


def get_generator(generator_name):
    """Return the generator function for the given instruction mode.

    :param generator_name: Name of the instruction mode, e.g. 'flat'
    :return: generator function
    """
    generator_func = eval("saltax.generator_" + generator_name)
    return generator_func


def xenonnt_salted_fuse(
    runid=None,
    saltax_mode="salt",
    output_folder="./fuse_data",
    cut_list=cutax.BasicCuts,
    corrections_version=DEFAULT_XEDOCS_VERSION,
    simulation_config_file="fuse_config_nt_sr1_dev.json",
    run_id_specific_config={
        "gain_model_mc": "gain_model",
        "electron_lifetime_liquid": "elife",
        "drift_velocity_liquid": "electron_drift_velocity",
        "drift_time_gate": "electron_drift_time_gate",
    },
    run_without_proper_corrections=False,
    generator_name="flat",
    recoil=8,
    simu_mode="all",
    **kwargs,
):
    """Return a strax context for XENONnT data analysis with saltax.

    :param runid: run number in integer. Must exist in RunDB if you use
        this context to compute raw_records_simu, or use None for data-
        loading only.
    :param saltax_mode: 'data', 'simu', or 'salt'.
    :param output_folder: Directory where data will be stored, defaults
        to ./strax_data
    :param corrections_version: XENONnT documentation version to use,
        defaults to DEFAULT_XEDOCS_VERSION
    :param cut_list: Cut list to use, defaults to cutax.BasicCuts
    :param simulation_config_file: File containing simulation
        configuration
    :param run_id_specific_config: Mapping of run_id specific config
    :param run_without_proper_corrections: Whether to run without proper
        corrections, defaults to False
    :param generator_name: Instruction mode to use, defaults to 'flat'
    :param recoil: NEST recoil type, defaults to 8
    :param simu_mode: 's1', 's2', or 'all'. Defaults to 'all'
    :param kwargs: Extra options to pass to strax.Context or generator
    :return: strax context
    """
    # Manually assign a correction_run_id if runid is None
    if runid is None:
        corrections_run_id = CORRECTION_RUN_ID_DEFAULT
    else:
        corrections_run_id = runid
    if (corrections_version is None) & (not run_without_proper_corrections):
        raise ValueError(
            "Specify a corrections_version. If you want to run without proper "
            "corrections for testing or just trying out fuse, "
            "set run_without_proper_corrections to True"
        )
    if simulation_config_file is None:
        raise ValueError("Specify a simulation configuration file")

    if run_without_proper_corrections:
        log.warning(
            "Running without proper correction version. This is not recommended for production use."
            "Take the context defined in cutax if you want to run XENONnT simulations."
        )

    st = strax.Context(storage=strax.DataDirectory(output_folder), **XNT_COMMON_OPTS)

    st.config.update(
        dict(
            # detector='XENONnT',
            check_raw_record_overlaps=True,
            **XNT_COMMON_OPTS,
            **FXNT_COMMON_CONFIG,
        )
    )

    for plugin_list in FUSED_PLUGINS:
        for plugin in plugin_list:
            st.register(plugin)

    if corrections_version is not None:
        st.apply_xedocs_configs(version=corrections_version)

    fuse.context.set_simulation_config_file(st, simulation_config_file)

    local_versions = st.config
    for config_name, url_config in local_versions.items():
        if isinstance(url_config, str):
            if "run_id" in url_config:
                local_versions[config_name] = straxen.URLConfig.format_url_kwargs(
                    url_config, run_id=corrections_run_id
                )
    st.config = local_versions

    # Update some run specific config
    for mc_config, processing_config in run_id_specific_config.items():
        if processing_config in st.config:
            st.config[mc_config] = st.config[processing_config]
        else:
            print(f"Warning! {processing_config} not in context config, skipping...")

    # No blinding in simulations
    st.config["event_info_function"] = "disabled"

    # Deregister plugins with missing dependencies
    st.deregister_plugins_with_missing_dependencies()

    # Add saltax mode
    st.set_config(dict(saltax_mode=saltax_mode))

    # Register cuts plugins
    if cut_list is not None:
        st.register_cut_list(cut_list)

    # Get salt generator
    generator_func = get_generator(generator_name)

    # Specify simulation instructions
    instr_file_name = saltax.instr_file_name(
        runid=runid, recoil=recoil, generator_name=generator_name, mode=simu_mode, **kwargs
    )
    # If runid is not None, then we need to either load instruction or generate it
    if runid is not None:
        # Try to load instruction from file and generate if not found
        try:
            instr = pd.read_csv(instr_file_name)
            print("Loaded instructions from file", instr_file_name)
        except:
            print(f"Instruction file {instr_file_name} not found. Generating instructions...")
            instr = generator_func(runid=runid, **kwargs)
            pd.DataFrame(instr).to_csv(instr_file_name, index=False)
            print(f"Instructions saved to {instr_file_name}")

        # Load instructions into config
        st.set_config(
            {
                "input_file": instr_file_name,
                "raw_records_file_size_target": MAX_RAW_RECORDS_FILE_SIZE_MB,
            }
        )
    
    # register the csv input plugin
    st.register(saltax.SChunkCsvInput)

    return st


def xenonnt_salted_wfsim(
    runid=None,
    saltax_mode="salt",
    output_folder="./strax_data",
    corrections_version=DEFAULT_XEDOCS_VERSION,
    cut_list=cutax.BasicCuts,
    auto_register_cuts=True,
    faxconf_version="sr0_v4",
    cmt_version="global_v9",
    cmt_run_id="026000",
    generator_name="flat",
    recoil=8,
    simu_mode="all",
    **kwargs,
):
    """Return a strax context for XENONnT data analysis with saltax.

    :param runid: run number in integer. Must exist in RunDB if you use
        this context to compute raw_records_simu, or use None for data-
        loading only.
    :param saltax_mode: 'data', 'simu', or 'salt'.
    :param output_folder: Directory where data will be stored, defaults
        to ./strax_data
    :param corrections_version: XENONnT documentation version to use,
        defaults to DEFAULT_XEDOCS_VERSION
    :param cut_list: Cut list to use, defaults to cutax.BasicCuts
    :param auto_register_cuts: Whether to automatically register cuts,
        defaults to True
    :param faxconf_version: (for simulation) fax configuration version
        to use, defaults to "sr0_v4"
    :param cmt_version: (for simulation) CMT version to use, defaults to
        "global_v9"
    :param cmt_run_id: (for simulation) CMT run ID to use, defaults to
        "026000"
    :param generator_name: (for simulation) Instruction mode to use,
        defaults to 'flat'
    :param recoil: (for simulation) NEST recoil type, defaults to 7
        (beta ER)
    :param simu_mode: 's1', 's2', or 'all'. Defaults to 'all'
    :param kwargs: Extra options to pass to strax.Context or generator,
        and rate for generator
    :return: strax context
    """
    # Get salt generator
    generator_func = get_generator(generator_name)

    # Specify simulation instructions
    instr_file_name = saltax.instr_file_name(
        runid=runid, recoil=recoil, generator_name=generator_name, mode=simu_mode, **kwargs
    )
    # If runid is not None, then we need to either load instruction or generate it
    if runid is not None:
        try:
            instr = pd.read_csv(instr_file_name)
            print("Loaded instructions from file", instr_file_name)
        except:
            print(f"Instruction file {instr_file_name} not found. Generating instructions...")
            instr = generator_func(runid=runid, **kwargs)
            pd.DataFrame(instr).to_csv(instr_file_name, index=False)
            print(f"Instructions saved to {instr_file_name}")

    # Based on cutax.xenonnt_sim_base()
    fax_conf = "fax_config_nt_{:s}.json".format(faxconf_version)

    # Based on straxen.contexts.xenonnt_online()
    if kwargs is not None:
        context_options = dict(**SXNT_COMMON_OPTS, **kwargs)
    else:
        context_options = SXNT_COMMON_OPTS.copy()
    context_config = dict(
        detector="XENONnT",  # from straxen.contexts.xenonnt_simulation()
        fax_config=fax_conf,  # from straxen.contexts.xenonnt_simulation()
        check_raw_record_overlaps=True,  # from straxen.contexts.xenonnt_simulation()
        **SXNT_COMMON_CONFIG,
    )
    st = strax.Context(
        storage=strax.DataDirectory(output_folder), config=context_config, **context_options
    )
    st.register([straxen.DAQReader, saltax.SRawRecordsFromFaxNT, saltax.SPeaklets])
    st.deregister_plugins_with_missing_dependencies()

    # Based on straxen.contexts.xenonnt()
    # st.apply_cmt_version(cmt_version)
    if corrections_version is not None:
        st.apply_xedocs_configs(version=corrections_version, **kwargs)

    # Based on cutax.xenonnt_offline()
    # extra plugins to register
    st.set_config(
        {
            "event_info_function": "blinding_v11",
            "avg_se_gain": "bodega://se_gain?bodega_version=v1",
            "g1": "bodega://g1?bodega_version=v5",
            "g2": "bodega://g2?bodega_version=v5",
        }
    )
    if auto_register_cuts:
        st.register_cuts()
    if cut_list is not None:
        st.register_cut_list(cut_list)

    # Based on straxen.xenonnt_simulation()
    _config_overlap = immutabledict(
        drift_time_gate="electron_drift_time_gate",
        drift_velocity_liquid="electron_drift_velocity",
        electron_lifetime_liquid="elife",
    )
    if straxen.utilix_is_configured(
        warning_message="Bad context as we cannot set CMT since we " "have no database access" ""
    ):
        st.apply_cmt_version(cmt_version)
    # Replace default cmt options with cmt_run_id tag + cmt run id
    cmt_options_full = straxen.get_corrections.get_cmt_options(st)
    # prune to just get the strax options
    cmt_options = {key: val["strax_option"] for key, val in cmt_options_full.items()}
    # First, fix gain model for simulation
    # Using placeholders for gain_model_mc
    st.set_config({"gain_model_mc": ("cmt_run_id", cmt_run_id, "legacy-to-pe://to_pe_placeholder")})
    fax_config_override_from_cmt = dict()
    for fax_field, cmt_field in _config_overlap.items():
        value = cmt_options[cmt_field]
        # URL configs need to be converted to the expected format
        if isinstance(value, str):
            opt_cfg = cmt_options_full[cmt_field]
            version = straxen.URLConfig.kwarg_from_url(value, "version")
            # We now allow the cmt name to be different from the config name
            # WFSim expects the cmt name
            value = (opt_cfg["correction"], version, True)
        fax_config_override_from_cmt[fax_field] = ("cmt_run_id", cmt_run_id, *value)
    st.set_config({"fax_config_override_from_cmt": fax_config_override_from_cmt})
    # and all other parameters for processing
    for option in cmt_options:
        value = cmt_options[option]
        if isinstance(value, str):
            # for URL configs we can just replace the run_id keyword argument
            # This will become the proper way to override the run_id for cmt configs
            st.config[option] = straxen.URLConfig.format_url_kwargs(value, run_id=cmt_run_id)
        else:
            # FIXME: Remove once all cmt configs are URLConfigs
            st.config[option] = ("cmt_run_id", cmt_run_id, *value)

    # Load instructions
    st.set_config(dict(fax_file=instr_file_name))  # doesn't matter for lineage
    st.set_config(dict(saltax_mode=saltax_mode))

    # Register pema plugins
    # st.register_all(pema.match_plugins)

    return st


def fxenonnt(
    runid=None,
    saltax_mode="salt",
    output_folder="./fuse_data",
    cut_list=cutax.BasicCuts,
    corrections_version=DEFAULT_XEDOCS_VERSION,
    simulation_config_file="fuse_config_nt_sr1_dev.json",
    run_id_specific_config={
        "gain_model_mc": "gain_model",
        "electron_lifetime_liquid": "elife",
        "drift_velocity_liquid": "electron_drift_velocity",
        "drift_time_gate": "electron_drift_time_gate",
    },
    run_without_proper_corrections=False,
    generator_name="flat",
    recoil=8,
    simu_mode="all",
    **kwargs,
):
    """United strax context for XENONnT data, simulation, or salted data. Based
    on fuse.

    :param runid: run number in integer. Must exist in RunDB if you use
        this context to compute raw_records_simu, or use None for data-
        loading only.
    :param saltax_mode: 'data', 'simu', or 'salt'
    :param output_folder: Output folder for strax data, default
        './strax_data'
    :param corrections_version: xedocs version to use, default is synced
        with cutax latest
    :param cut_list: List of cuts to register, default is
        cutax.BasicCuts
    :param auto_register_cuts: Whether to auto register cuts, default
        True
    :param faxconf_version: fax config version to use, default is synced
        with cutax latest
    :param cmt_version: cmt version to use, default is synced with cutax
        latest
    :param cmt_run_id: cmt run id to use, default is synced with cutax
    :param generator_name: (for simulation) Instruction mode to use,
        defaults to 'flat'
    :param recoil: (for simulation) NEST recoil type, defaults to 7
        (beta ER)
    :param simu_mode: 's1', 's2', or 'all'. Defaults to 'all'
    :param kwargs: Additional kwargs to pass
    :return: strax context
    """
    assert saltax_mode in SALTAX_MODES, "saltax_mode must be one of %s" % (SALTAX_MODES)
    if runid is None:
        print(
            "Since you specified runid=None, this context will not be able to compute raw_records_simu."
        )
        print("Welcome to data-loading only mode!")
    else:
        print("Welcome to computation mode which only works for run %s!" % (runid))

    return xenonnt_salted_fuse(
        runid=runid,
        output_folder=output_folder,
        corrections_version=corrections_version,
        cut_list=cut_list,
        simulation_config_file=simulation_config_file,
        run_id_specific_config=run_id_specific_config,
        run_without_proper_corrections=run_without_proper_corrections,
        generator_name=generator_name,
        recoil=recoil,
        simu_mode=simu_mode,
        **kwargs,
    )


def sxenonnt(
    runid=None,
    saltax_mode="salt",
    output_folder="./strax_data",
    corrections_version=DEFAULT_XEDOCS_VERSION,
    cut_list=cutax.BasicCuts,
    auto_register_cuts=True,
    faxconf_version="sr0_v4",
    cmt_version="global_v9",
    cmt_run_id="026000",
    generator_name="flat",
    recoil=7,
    simu_mode="all",
    **kwargs,
):
    """United strax context for XENONnT data, simulation, or salted data. Based
    on wfsim.

    :param runid: run number in integer. Must exist in RunDB if you use
        this context to compute raw_records_simu, or use None for data-
        loading only.
    :param saltax_mode: 'data', 'simu', or 'salt'
    :param output_folder: Output folder for strax data, default
        './strax_data'
    :param corrections_version: xedocs version to use, default is synced
        with cutax latest
    :param cut_list: List of cuts to register, default is
        cutax.BasicCuts
    :param auto_register_cuts: Whether to auto register cuts, default
        True
    :param faxconf_version: fax config version to use, default is synced
        with cutax latest
    :param cmt_version: cmt version to use, default is synced with cutax
        latest
    :param cmt_run_id: cmt run id to use, default is synced with cutax
    :param generator_name: (for simulation) Instruction mode to use,
        defaults to 'flat'
    :param recoil: (for simulation) NEST recoil type, defaults to 7
        (beta ER)
    :param simu_mode: 's1', 's2', or 'all'. Defaults to 'all'
    :param kwargs: Additional kwargs to pass
    :return: strax context
    """
    assert saltax_mode in SALTAX_MODES, "saltax_mode must be one of %s" % (SALTAX_MODES)
    if runid is None:
        print(
            "Since you specified runid=None, this context will not be able to compute raw_records_simu."
        )
        print("Welcome to data-loading only mode!")
    else:
        print("Welcome to computation mode which only works for run %s!" % (runid))

    return xenonnt_salted_wfsim(
        runid=runid,
        output_folder=output_folder,
        corrections_version=corrections_version,
        cut_list=cut_list,
        auto_register_cuts=auto_register_cuts,
        faxconf_version=faxconf_version,
        cmt_version=cmt_version,
        cmt_run_id=cmt_run_id,
        generator_name=generator_name,
        recoil=recoil,
        simu_mode=simu_mode,
        saltax_mode=saltax_mode,
        **kwargs,
    )
