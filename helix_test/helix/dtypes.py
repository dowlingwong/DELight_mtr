import strax
import numpy as np

# Don't forget to add any added entity into this list to make it importable on import helix
__all__ = ['time_dt_fields', 'triggers_dtype', 'get_raw_records_dtype', 'get_events_dtype', 'get_fit_results_dtype',
           'get_psds_dtype', ]

# same as strax.time_dt_fields, but with int64 instead of int32 for the length field, to allow for long data kinds
time_dt_fields = [
    (('Start time since unix epoch [ns]', 'time'), np.int64),
    # In strax.time_dt_fields, length is int32. We change it to int64 to allow for long data kinds
    (('Length of the interval in samples', 'length'), np.int64),
    (('Width of one sample [ns]', 'dt'), np.int16),
]

# dtype for triggers
triggers_dtype = time_dt_fields + [
    (('Channel number', 'channel'), np.int16),
    (('Channel type', 'channel_type'), np.int16),
    (('Id of the block of records in the run', 'block_id'), np.int32),
    (('Sample index in the record where the filtered record crossed the threshold', 'start_loc'), np.int32),
    (('Sample index in the record where the filtered record reached the maximum', 'loc'), np.int32),
    (('Max amplitude of the filtered record withing the trigger', 'amplitude'), np.float32),
    (('Whether or not the deactivation threshold was crossed', 'deactivation_crossed'), np.bool_),
]


def get_raw_records_dtype(record_length):
    """
    Returns the dtype for raw_records, with the provided record length
    :param record_length: length of records in time samples
    :return: dtype for raw records
    """
    return time_dt_fields + [
        (('Channel number', 'channel'), np.int16),
        (('Channel type', 'channel_type'), np.int16),
        (('Id of the block of records in the run', 'block_id'), np.int32),
        (('Record data in raw ADC counts', 'data'), np.int16, record_length),
    ]


def get_events_dtype(event_length, n_channels, n_summed_channels):
    """
    Returns the dtype for structured arrays of events
    :param event_length: event length in samples
    :param n_channels: number of channels
    :param n_summed_channels: number of summed channel types
    :return: dtype for events
    """
    return time_dt_fields + [
        (('Id of the block of records in the run', 'block_id'), np.int32),
        (('Start time of the block since unix epoch [ns]', 'block_time'), np.int64),
        (('Event number in the record block', 'event_id'), np.int16),
        (('Sample index in the record where the trace starts', 'start_loc'), np.int32),
        (('Trace data in individual channels', 'channel_data'), np.float64, (n_channels, event_length)),
        (('Channel numbers', 'channels'), np.int16, n_channels),
        (('Summed traces of summed_channel_types', 'data'), np.float64, (n_summed_channels, event_length)),
        (('Types of the summed traces', 'summed_channel_types'), np.int16, n_summed_channels),
        (('Mask of channels that were summed up to produce the summed traces', 'summed_channel_masks'), np.bool_,
         (n_summed_channels, n_channels)),
    ]


def get_fit_results_dtype(n_submerged_channels, n_vacuum_channels):
    """
    Returns the dtype for structured arrays of events
    :param n_submerged_channels: number of submerged channels
    :param n_vacuum_channels: number of vacuum channels
    :return: dtype for fit_results
    """
    n_channels = n_submerged_channels + n_vacuum_channels
    return time_dt_fields + [
        (('Id of the block of records in the run', 'block_id'), np.int32),
        (('Event number in the record block', 'event_id'), np.int16),
        (('Channel numbers', 'channels'), np.int16, n_channels),

        (('Amplitudes of OF UV fits in individual submerged channels',
          'submerged_channel_uv_amplitude'), np.float64, n_submerged_channels),
        (('Chi-squared values of OF UV fits in individual submerged channels',
          'submerged_channel_fit_chi2'), np.float64, n_submerged_channels),
        (('Time offsets of UV template in the OF fits in individual submerged channels in samples',
          'submerged_channel_uv_offset'), np.int32, n_submerged_channels),

        (('OF UV fit amplitude in the sum of submerged channels',
          'submerged_sum_uv_amplitude'), np.float64),
        (('OF UV fit chi-squared value in sum of submerged channels',
          'submerged_sum_fit_chi2'), np.float64),
        (('OF UV fit template offset in the sum of submerged channels in samples',
          'submerged_sum_uv_offset'), np.int32),

        (('OF UV fit amplitude in the sum of triggered submerged channels',
          'submerged_triggered_uv_amplitude'), np.float64),
        (('OF UV fit chi-squared value in the sum of triggered submerged channels',
          'submerged_triggered_fit_chi2'), np.float64),
        (('OF UV fit template offset in the sum of triggered submerged channels in samples',
          'submerged_triggered_uv_offset'), np.int32),
        (('Mask of triggered submerged channels',
          'submerged_triggered_channel_masks'), np.bool_, n_channels),

        (('UV amplitudes of 2-template OF fits in individual vacuum channels',
          'vacuum_channel_uv_amplitude'), np.float64, n_vacuum_channels),
        (('QP amplitudes of 2-template OF fits in individual vacuum channels',
          'vacuum_channel_qp_amplitude'), np.float64, n_vacuum_channels),
        (('Chi-squared values of 2-template OF fits in individual vacuum channels',
          'vacuum_channel_fit_chi2'), np.float64, n_vacuum_channels),
        (('UV template time shifts in 2-template OF fits in individual vacuum channels in samples',
          'vacuum_channel_uv_offset'), np.int32, n_vacuum_channels),
        (('QP template time shifts in 2-template OF fits in individual vacuum channels in samples',
          'vacuum_channel_qp_offset'), np.int32, n_vacuum_channels),

        (('UV amplitude of 2-template OF fits in the sum of vacuum channels',
          'vacuum_sum_uv_amplitude'), np.float64),
        (('QP amplitude of 2-template OF fits in the sum of vacuum channels',
          'vacuum_sum_qp_amplitude'), np.float64),
        (('2-template OF fit chi-squared value in the sum of vacuum channels',
          'vacuum_sum_fit_chi2'), np.float64),
        (('UV template time shift in the 2-template OF fit in the sum of vacuum channels in samples',
          'vacuum_sum_uv_offset'), np.int32),
        (('QP template time shift in the 2-template OF fit in the sum of vacuum channels in samples',
          'vacuum_sum_qp_offset'), np.int32),

        (('UV amplitude of 2-template OF fits in the sum of triggered vacuum channels',
          'vacuum_triggered_uv_amplitude'), np.float64),
        (('QP amplitude of 2-template OF fits in the sum of triggered vacuum channels',
          'vacuum_triggered_qp_amplitude'), np.float64),
        (('2-template OF fit chi-squared value in the sum of triggered vacuum channels',
          'vacuum_triggered_fit_chi2'), np.float64),
        (('UV template time shift in the 2-template OF fit in the sum of triggered vacuum channels in samples',
          'vacuum_triggered_uv_offset'), np.int32),
        (('QP template time shift in the 2-template OF fit in the sum of triggered vacuum channels in samples',
          'vacuum_triggered_qp_offset'), np.int32),
        (('Mask of triggered vacuum channels',
          'vacuum_triggered_channel_masks'), np.bool_, n_channels),

        (('UV amplitude of 2-template OF fits in the sum of all triggered channels',
          'triggered_uv_amplitude'), np.float64),
        (('QP amplitude of 2-template OF fits in the sum of all triggered channels',
          'triggered_qp_amplitude'), np.float64),
        (('2-template OF fit chi-squared value in the sum of all triggered channels',
          'triggered_fit_chi2'), np.float64),
        (('UV template time shift in the 2-template OF fit in the sum of all triggered channels in samples',
          'triggered_uv_offset'), np.int32),
        (('QP template time shift in the 2-template OF fit in the sum of all triggered channels in samples',
          'triggered_qp_offset'), np.int32),
        (('Mask of all triggered channels',
          'triggered_channel_masks'), np.bool_, n_channels),

        (('UV amplitude of 2-template OF fits in the sum of all channels',
          'sum_uv_amplitude'), np.float64),
        (('QP amplitude of 2-template OF fits in the sum of all channels',
          'sum_qp_amplitude'), np.float64),
        (('2-template OF fit chi-squared value in the sum of all channels',
          'sum_fit_chi2'), np.float64),
        (('UV template time shift in the 2-template OF fit in the sum of all channels in samples',
          'sum_uv_offset'), np.int32),
        (('QP template time shift in the 2-template OF fit in the sum of all channels in samples',
          'sum_qp_offset'), np.int32),
    ]


def get_psds_dtype(noise_event_length, n_channels, n_summed_channels):
    """
    Returns the dtype for structured arrays of noise PSDs
    :param noise_event_length: length of noise events
    :param n_channels: number of channels
    :param n_summed_channels: number of summed channel types
    :return: dtype for noise PSDs
    """
    psd_length = noise_event_length // 2 + 1 if noise_event_length % 2 == 0 else (noise_event_length + 1) // 2
    n_csds = n_channels * (n_channels - 1) // 2
    return strax.time_fields + [
        (('Number of noise events used to calculate the PSDs', 'n_events'), np.int32),
        (('Noise PSDs in ADC^2/Hz', 'psds'), np.float64, (n_channels, psd_length)),
        (('Noise CSDs in ADC^2/Hz', 'csds'), np.float64, (n_csds, psd_length)),
        (('PSD frequencies in Hz', 'frequencies'), np.float64, psd_length),
        (('Channel numbers', 'channels'), np.int16, n_channels),
        (('Noise PSDs of summed channels in ADC^2/Hz', 'summed_channel_psds'), np.float64,
         (n_summed_channels, psd_length)),
        (('Types of the summed channels', 'summed_channel_types'), np.int16, n_summed_channels),
    ]
