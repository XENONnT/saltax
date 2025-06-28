import straxen
import saltax
import cutax
import strax
from immutabledict import immutabledict
import fuse
import logging
import pandas as pd
from utilix import xent_collection


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.context")


# fuse plugins
# Plugins to simulate microphysics
MICROPHYSICS_PLUGINS_DBSCAN_CLUSTERING = fuse.context.microphysics_plugins_dbscan_clustering
MICROPHYSICS_PLUGINS_LINEAGE_CLUSTERING = fuse.context.microphysics_plugins_lineage_clustering
REMAINING_MICROPHYSICS_PLUGINS = fuse.context.remaining_microphysics_plugins
# Plugins to simulate S1 signals
S1_SIMULATION_PLUGINS = fuse.context.s1_simulation_plugins
# Plugins to simulate S2 signals
S2_SIMULATION_PLUGINS = fuse.context.s2_simulation_plugins
# Plugins to simulate delayed electrons
DELAYED_ELECTRON_SIMULATION_PLUGINS = fuse.context.delayed_electron_simulation_plugins
# Plugins to merge delayed and regular electrons
DELAYED_ELECTRON_MERGER_PLUGINS = fuse.context.delayed_electron_merger_plugins
# Plugins to simulate PMTs and DAQ
PMT_AND_DAQ_PLUGINS = fuse.context.pmt_and_daq_plugins
# Plugins to get truth information
TRUTH_INFORMATION_PLUGINS = fuse.context.truth_information_plugins
# All plugins with fuse
FUSED_PLUGINS = [
    MICROPHYSICS_PLUGINS_DBSCAN_CLUSTERING,
    MICROPHYSICS_PLUGINS_LINEAGE_CLUSTERING,
    REMAINING_MICROPHYSICS_PLUGINS,
    S1_SIMULATION_PLUGINS,
    S2_SIMULATION_PLUGINS,
    DELAYED_ELECTRON_SIMULATION_PLUGINS,
    DELAYED_ELECTRON_MERGER_PLUGINS,
    PMT_AND_DAQ_PLUGINS,
    TRUTH_INFORMATION_PLUGINS,
]
FUSE_DONT_REGISTER = [
    fuse.plugins.micro_physics.microphysics_summary.MicroPhysicsSummary,
    fuse.plugins.pmt_and_daq.pmt_response_and_daq.PMTResponseAndDAQ,
]

# ~Infinite raw_records file size to avoid downchunking
MAX_RAW_RECORDS_FILE_SIZE_MB = 1e9

# straxen XENONnT options/configuration
XNT_COMMON_OPTS = straxen.contexts.xnt_common_opts.copy()
XNT_COMMON_CONFIG = straxen.contexts.xnt_common_config.copy()

# fuse based saltax options overrides
FXNT_COMMON_OPTS_REGISTER = XNT_COMMON_OPTS["register"].copy()
FXNT_COMMON_OPTS_REGISTER.remove(straxen.Peaklets)
FXNT_COMMON_OPTS_REGISTER.remove(straxen.PulseProcessing)
FXNT_COMMON_OPTS_REGISTER = [
    saltax.SChunkCsvInput,
    saltax.SPeaklets,
    saltax.SPulseProcessing,
    saltax.SPMTResponseAndDAQ,
] + FXNT_COMMON_OPTS_REGISTER
FXNT_COMMON_OPTS_OVERRIDE = dict(
    register=FXNT_COMMON_OPTS_REGISTER,
)
FXNT_COMMON_OPTS = XNT_COMMON_OPTS.copy()
FXNT_COMMON_OPTS["register"] = FXNT_COMMON_OPTS_OVERRIDE["register"]

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
FXNT_COMMON_CONFIG = XNT_COMMON_CONFIG.copy()
FXNT_COMMON_CONFIG["channel_map"] = XNT_COMMON_CONFIG_OVERRIDE["channel_map"]
DEFAULT_XEDOCS_VERSION = cutax.contexts.DEFAULT_XEDOCS_VERSION

# saltax modes supported
SALTAX_MODES = ["data", "simu", "salt"]


def validate_runid(runid):
    """Validate runid in RunDB to see if you can use it for computation.

    :param runid: run number in integer
    :return: None
    """
    try:
        doc = xent_collection().find_one({"number": runid})
        if doc is None:
            raise ValueError(f"Run {runid} not found in RunDB")
    except Exception as e:
        raise ValueError(f"Run {runid} not found in RunDB: {e}")


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
    simu_config_version="sr1_dev",
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
    :param simu_config_version: simulation configuration version to use,
        defaults to "sr1_dev"
    :param run_id_specific_config: Mapping of run_id specific config
    :param run_without_proper_corrections: Whether to run without proper
        corrections, defaults to False
    :param generator_name: Instruction mode to use, defaults to 'flat'
    :param recoil: NEST recoil type, defaults to 8
    :param simu_mode: 's1', 's2', or 'all'. Defaults to 'all'
    :param kwargs: Extra options to pass to strax.Context or generator
    :return: strax context
    """
    if (corrections_version is None) & (not run_without_proper_corrections):
        raise ValueError(
            "Specify a corrections_version. If you want to run without proper "
            "corrections for testing or just trying out fuse, "
            "set run_without_proper_corrections to True"
        )
    if simu_config_version is None:
        raise ValueError("Specify a simulation configuration file")

    if run_without_proper_corrections:
        log.warning(
            "Running without proper correction version. This is not recommended for production use."
            "Take the context defined in cutax if you want to run XENONnT simulations."
        )

    if kwargs is not None:
        context_options = dict(**FXNT_COMMON_OPTS, **kwargs)
    else:
        context_options = FXNT_COMMON_OPTS.copy()
    context_config = dict(
        check_raw_record_overlaps=True,
        **FXNT_COMMON_CONFIG,
    )
    st = strax.Context(
        storage=strax.DataDirectory(output_folder), config=context_config, **context_options
    )

    for plugin_list in FUSED_PLUGINS:
        for plugin in plugin_list:
            if plugin not in FUSE_DONT_REGISTER:
                st.register(plugin)

    if corrections_version is not None:
        st.apply_xedocs_configs(version=corrections_version)

    simulation_config_file = "fuse_config_nt_{:s}.json".format(simu_config_version)
    fuse.context.set_simulation_config_file(st, simulation_config_file)

    # Update some run specific config
    for mc_config, processing_config in run_id_specific_config.items():
        if processing_config in st.config:
            st.config[mc_config] = st.config[processing_config]
        else:
            log.warning(f"Warning! {processing_config} not in context config, skipping...")

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
            log.info("Loaded instructions from file", instr_file_name)
        except:
            log.info(f"Instruction file {instr_file_name} not found. Generating instructions...")
            instr = generator_func(runid=runid, recoil=recoil, **kwargs)
            pd.DataFrame(instr).to_csv(instr_file_name, index=False)
            log.info(f"Instructions saved to {instr_file_name}")

        # Load instructions into config
        st.set_config(
            {
                "input_file": instr_file_name,
                "raw_records_file_size_target": MAX_RAW_RECORDS_FILE_SIZE_MB,
            }
        )

    return st


def fxenonnt(
    runid=None,
    saltax_mode="salt",
    output_folder="./fuse_data",
    cut_list=cutax.BasicCuts,
    corrections_version=DEFAULT_XEDOCS_VERSION,
    simu_config_version="sr1_dev",
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
    unblind=True,
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
    :param simu_config_version: fax config version to use, default is
        synced with cutax latest
    :param run_id_specific_config: Mapping of run_id specific config
    :param run_without_proper_corrections: Whether to run without proper
        corrections, defaults to False
    :param generator_name: Instruction mode to use, defaults to 'flat'
    :param recoil: NEST recoil type, defaults to 8 (beta ER)
    :param simu_mode: 's1', 's2', or 'all'. Defaults to 'all'
    :param unblind: Whether to bypass any kind of blinding, defaults to True
    :param kwargs: Extra options to pass to strax.Context or generator,
        and rate/en_range etc for generator
    :return: strax context
    """
    assert saltax_mode in SALTAX_MODES, "saltax_mode must be one of %s" % (SALTAX_MODES)
    if runid is None:
        log.warning(
            "Since you specified runid=None, this context will not be able to compute raw_records_simu."
        )
        log.warning("Welcome to data-loading only mode!")
    else:
        validate_runid(runid)
        log.warning("Welcome to computation mode which only works for run %s!" % (runid))

    st = xenonnt_salted_fuse(
        runid=runid,
        saltax_mode=saltax_mode,
        output_folder=output_folder,
        corrections_version=corrections_version,
        cut_list=cut_list,
        simu_config_version=simu_config_version,
        run_id_specific_config=run_id_specific_config,
        run_without_proper_corrections=run_without_proper_corrections,
        generator_name=generator_name,
        recoil=recoil,
        simu_mode=simu_mode,
        **kwargs,
    )
    if unblind:
        st.set_config(dict(event_info_function="disabled"))

    return st
