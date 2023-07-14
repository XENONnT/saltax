import straxen
from immutabledict import immutabledict

# XENONnT common configuration
COMMON_OPTS = straxen.contexts.common_opts
XNT_COMMON_CONFIG = straxen.contexts.xnt_common_config

# saltax overrides
SALT_CHANNEL_START = 3000 # Channel number for the first salted channel
COMMON_OPTS_OVERRIDE = dict(
    register_all=[],
    # Register all peak/pulse processing by hand as 1T does not need to have
    # the high-energy plugins.
    register=[
        straxen.PulseProcessing, # TODO: to be replaced
        straxen.Peaklets,
        straxen.PeakletClassification,
        straxen.MergedS2s,
        straxen.Peaks,
        straxen.PeakBasics,
        straxen.PeakProximity,
        straxen.Events,
        straxen.EventBasics,
        straxen.EventPositions,
        straxen.CorrectedAreas,
        straxen.EnergyEstimates,
        straxen.EventInfoDouble,
        straxen.DistinctChannels,
    ],
)
XNT_COMMON_CONFIG_OVERRIDE = dict(
    channel_map=immutabledict(
        # (Minimum channel, maximum channel)
        # Channels must be listed in a ascending order!
        tpc=(0, 493),
        he=(500, 752),  # high energy
        aqmon=(790, 807),
        aqmon_nv=(808, 815),  # nveto acquisition monitor
        tpc_blank=(999, 999),
        mv=(1000, 1083),
        aux_mv=(1084, 1087),  # Aux mv channel 2 empty  1 pulser  and 1 GPS
        mv_blank=(1999, 1999),
        nveto=(2000, 2119),
        nveto_blank=(2999, 2999),
        stpc=(SALT_CHANNEL_START, SALT_CHANNEL_START+493) # Salted TPC channels
    ),
)
SCOMMON_OPTS = COMMON_OPTS.update(COMMON_OPTS_OVERRIDE)
SXNT_COMMON_CONFIG = COMMON_OPTS.update(XNT_COMMON_CONFIG_OVERRIDE)
