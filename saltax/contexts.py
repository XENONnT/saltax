import straxen
import saltax
import cutax
from immutabledict import immutabledict
from cutax.cut_lists.basic import BasicCuts

# straxen XENONnT options/configuration
XNT_COMMON_OPTS = straxen.contexts.xnt_common_opts
XNT_COMMON_CONFIG = straxen.contexts.xnt_common_config
XNT_SIMULATION_CONFIG = straxen.contexts.xnt_simulation_config

# saltax options overrides
SXNT_COMMON_OPTS_REGISTER = XNT_COMMON_OPTS['register'].copy()
SXNT_COMMON_OPTS_REGISTER.remove(straxen.PulseProcessing)
SXNT_COMMON_OPTS_REGISTER = [saltax.PulseProcessing] + SXNT_COMMON_OPTS_REGISTER
XNT_COMMON_OPTS_OVERRIDE = dict(
    register=SXNT_COMMON_OPTS_REGISTER,
)
SXNT_COMMON_OPTS = XNT_COMMON_OPTS.update(XNT_COMMON_OPTS_OVERRIDE)

# saltax configuration overrides
SCHANNEL_STARTS_AT = -494
XNT_COMMON_CONFIG_OVERRIDE = dict(
    channel_map=immutabledict(
        # (Minimum channel, maximum channel)
        # Channels must be listed in a ascending order!
        stpc=(SCHANNEL_STARTS_AT, SCHANNEL_STARTS_AT+493), # Salted TPC channels
        tpc=(0, 493),
        he=(500, 752),  # high energy
        aqmon=(790, 807),
        aqmon_nv=(808, 815),  # nveto acquisition monitor
        tpc_blank=(999, 999),
        mv=(1000, 1083),
        aux_mv=(1084, 1087),  # Aux mv channel 2 empty  1 pulser  and 1 GPS
        mv_blank=(1999, 1999),
        nveto=(2000, 2119),
        nveto_blank=(2999, 2999)
    ),
)
SXNT_COMMON_CONFIG = XNT_COMMON_CONFIG.update(XNT_COMMON_CONFIG_OVERRIDE)

# saltax modes supported
SALTAX_MODES = ['data', 'simu', 'salt']

# cutax XENONnT contexts
XENONNT_OFFLINE = cutax.contexts.xenonnt_offline
XENONNT_SIMULATION = cutax.contexts.xenonnt_sim_base
DEFAULT_XEDOCS_VERSION = cutax.contexts.DEFAULT_XEDOCS_VERSION


def xenonnt_salted(xedocs_version=DEFAULT_XEDOCS_VERSION,
                   cut_list=BasicCuts, auto_register=True,
                   faxconf_version="sr0_v4",
                   cmt_version="global_v11",
                   wfsim_registry='RawRecordsFromFaxNT',
                   cmt_run_id="026000",
                   latest="sr0_v4",
                   simulate_nv=False,
                   **kwargs):
    return 'hi'


def sxenonnt(saltax_mode,
             xedocs_version=DEFAULT_XEDOCS_VERSION,
             cut_list=BasicCuts, auto_register=True,
             faxconf_version="sr0_v4",
             cmt_version="global_v11",
             wfsim_registry='RawRecordsFromFaxNT',
             cmt_run_id="026000",
             latest="sr0_v4",
             simulate_nv=False,
             **kwargs):
    assert saltax_mode in SALTAX_MODES, "saltax_mode must be one of %s"%(SALTAX_MODES)
    
    if saltax_mode == 'data':
        return XENONNT_OFFLINE(
            xedocs_version=xedocs_version,
            cut_list=cut_list,
            auto_register=auto_register,
            **kwargs)
    elif saltax_mode == 'simu':
        return XENONNT_SIMULATION(
            faxconf_version=faxconf_version,
            cmt_version=cmt_version,
            wfsim_registry=wfsim_registry,
            cmt_run_id=cmt_run_id,
            latest=latest,
            simulate_nv=simulate_nv,
            cut_list=cut_list,
            **kwargs)
    
