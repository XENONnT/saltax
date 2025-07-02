import logging
import inspect
from immutabledict import immutabledict
import pandas as pd

from utilix import xent_collection
import straxen
from fuse.context import full_chain_context, xenonnt_fuse_full_chain_simulation
import saltax
from saltax.instructions.generator import instr_file_name
from saltax.plugins.records import SCHANNEL_STARTS_AT

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
log = logging.getLogger("saltax.contexts")


# ~Infinite raw_records file size to avoid downchunking
MAX_RAW_RECORDS_FILE_SIZE_MB = 1e9

# saltax modes supported
SALTAX_MODES = ["data", "simu", "salt"]


def validate_runid(runid):
    """Validate runid in RunDB to see if you can use it for computation.

    :param runid: run number
    :return: None

    """
    doc = xent_collection().find_one({"number": int(runid)})
    if doc is None:
        raise ValueError(f"Run {runid} not found in RunDB")


def sxenonnt(
    runid=None,
    saltax_mode="salt",
    generator_name="flat",
    recoil=8,
    simu_mode="all",
    context=xenonnt_fuse_full_chain_simulation,
    cut_list=None,
    unblind=True,
    **kwargs,
):
    """Return a strax context for XENONnT data analysis with saltax.

    :param runid: run number. Must exist in RunDB if you use this context to compute
        raw_records_simu, or use None for data- loading only.
    :param saltax_mode: 'data', 'simu', or 'salt'.
    :param generator_name: Instruction mode to use (default: 'flat')
    :param recoil: NEST recoil type (default: 8)
    :param simu_mode: 's1', 's2', or 'all' (default: 'all')
    :param context: strax context to use (default: xenonnt_fuse_full_chain_simulation)
    :param cut_list: Cut list to use (default: None)
    :param unblind: Whether to bypass any kind of blinding (default: True)
    :param kwargs: Extra options to pass to strax.Context or generator
    :return: strax context

    """

    if runid is None:
        log.warning(
            "Since you specified runid=None, "
            "this context will not be able to compute raw_records_simu."
        )
        log.warning("Welcome to data-loading only mode!")
    else:
        validate_runid(runid)
        log.warning(f"Welcome to computation mode which only works for run {runid}!")

    if saltax_mode not in SALTAX_MODES:
        raise ValueError(f"saltax_mode must be one of {SALTAX_MODES} but got {saltax_mode}.")

    # Do not register cut_list if it is None and cutax is not installed
    if cut_list is None:
        try:
            import cutax

            _cut_list = cutax.BasicCuts
        except ImportError:
            log.warning("cutax is not installed, no cutlist will be registered.")
            _cut_list = None

    # xenonnt_fuse_full_chain_simulation has a kwargs as argument
    # and full_chain_context is called in it
    if context is xenonnt_fuse_full_chain_simulation:
        params = {
            **inspect.signature(context).parameters,
            **inspect.signature(full_chain_context).parameters,
        }
    else:
        params = inspect.signature(context).parameters
    _kwargs = {k: v for k, v in kwargs.items() if k in params}
    st = context(corrections_run_id=runid, cut_list=_cut_list, **_kwargs)

    # Register deregistered plugins when replacing DAQReader by PMTResponseAndDAQ
    st.register(
        (
            straxen.DAQReader,
            straxen.AqmonHits,
            straxen.VetoIntervals,
            straxen.VetoProximity,
        )
    )
    # Register saltax plugins
    st.register(
        (
            saltax.SChunkCsvInput,
            saltax.SPeaklets,
            saltax.SPulseProcessing,
            saltax.SPMTResponseAndDAQ,
        )
    )

    # Add saltax mode
    st.set_config({"saltax_mode": saltax_mode})

    # Modify channel map to include salted channels
    for key, cmap in st.config["channel_map"].items():
        if SCHANNEL_STARTS_AT <= cmap[1]:
            raise ValueError(
                f"Salted channels ({SCHANNEL_STARTS_AT}) must be after all possible channels "
                f"but {key} starts at {cmap[0]} and ends at {cmap[1]}"
            )
    channel_map = immutabledict(
        dict(
            stpc=(
                SCHANNEL_STARTS_AT,
                SCHANNEL_STARTS_AT + len(range(*st.config["channel_map"]["tpc"])) + 1,
            ),
            **st.config["channel_map"],
        )
    )
    st.set_config({"channel_map": channel_map})

    try:
        import cutax

        # Register cuts plugins
        for p in cutax.contexts.EXTRA_PLUGINS:
            st.register(p)
    except ImportError:
        pass

    # Deregister plugins with missing dependencies
    st.deregister_plugins_with_missing_dependencies()

    # Get salt generator
    generator_func = getattr(saltax.instructions.generator, "generator_" + generator_name)

    # Specify simulation instructions
    input_file = instr_file_name(
        runid=runid,
        recoil=recoil,
        generator_name=generator_name,
        mode=simu_mode,
        **straxen.filter_kwargs(instr_file_name, kwargs),
    )

    # If runid is not None, then we need to either load instruction or generate it
    if runid is not None:
        runid = str(runid).zfill(6)
        # Try to load instruction from file and generate if not found
        try:
            instr = pd.read_csv(input_file)
            log.info("Loaded instructions from file", input_file)
        except FileNotFoundError:
            log.info(f"Instruction file {input_file} not found. Generating instructions...")
            instr = generator_func(
                runid=runid,
                recoil=recoil,
                context=st,
                **straxen.filter_kwargs(generator_func, kwargs),
            )
            pd.DataFrame(instr).to_csv(input_file, index=False)
            log.info(f"Instructions saved to {input_file}")

        # Load instructions into config
        st.set_config(
            {
                "input_file": input_file,
                "raw_records_file_size_target": MAX_RAW_RECORDS_FILE_SIZE_MB,
            }
        )

    if unblind:
        st.set_config({"event_info_function": "disabled"})

    return st
