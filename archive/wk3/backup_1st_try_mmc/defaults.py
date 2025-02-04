from helix import units, ChannelType

# default values of all the configs
# it might make sense to keep separate default files for separate sets of plugins
# E.g., if we add a new set of plugins, let's say for MMC R&D analyses, we might want to create a separate file with
# default configs for these plugins

#This block has been adapted for MMC readout
DEFAULT_RECORD_LENGTH_NS = 8.4 * units.ms
DEFAULT_RECORD_LENGTH = 32_768  
DEFAULT_SAMPLING_DT = int(DEFAULT_RECORD_LENGTH_NS/DEFAULT_RECORD_LENGTH)
DEFAULT_SAMPLING_FREQUENCY = int(units.s/DEFAULT_SAMPLING_DT)

#This block has been adapted for MMC readout
DEFAULT_OF_LENGTH = 32_768
DEFAULT_TEMPLATE_LENGTH = 32_768
DEFAULT_PREPULSE_LENGTH = 3072

DEFAULT_ALLOWED_FIT_SHIFTS = (-200, 200)
DEFAULT_ALLOWED_TWO_TEMPLATE_DELTAS = (150, 350)
DEFAULT_FIT_SUMMED_TRIGGERED_CHANNELS = True

N_SUBMERGED_CHANNELS = 35
N_VACUUM_CHANNELS = 15

DEFAULT_CHANNEL_MAP = {
    ChannelType.SUBMERGED: (1, N_SUBMERGED_CHANNELS),
    ChannelType.VACUUM: (N_SUBMERGED_CHANNELS + 1, N_SUBMERGED_CHANNELS + N_VACUUM_CHANNELS)
}

DEFAULT_PRE_TRIGGER_NOISE_TRACE_VETO = 200
DEFAULT_POST_TRIGGER_NOISE_TRACE_VETO = int(0.2 * DEFAULT_SAMPLING_FREQUENCY)  # 0.2 sec

DEFAULT_N_NOISE_EVENTS_PER_RECORD = 2
DEFAULT_ALLOW_NOISE_EVENTS_OVERLAPS = False
DEFAULT_NOISE_EVENTS_RANDOM_SEED = None

DEFAULT_NOISE_PSD_DURATION = 1800 * units.s


helix_data_dir = 'test_helix_data'