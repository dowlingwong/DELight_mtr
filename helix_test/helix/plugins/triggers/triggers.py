import strax
import numpy as np
import helix as hx
import numba
from scipy import signal
from datetime import datetime

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
class Triggers(strax.Plugin):
    """
    Base plugin for triggering plugins that take raw records as input, convolve them with configurable filter kernel,
    and apply a double-threshold triggering with configurable activation and deactivation thresholds.
    """
    depends_on = 'raw_records'
    provides = 'triggers'
    data_kind = 'triggers'
    parallel = 'process'
    dtype = hx.triggers_dtype
    __version__ = '0.0.0'
    # TODO: test which compressions is the best one, blosc (default), or lz4, or something else?

    # TODO: move default values to the default.py file
    filter_kernel = strax.Config(
        default=None, type=list,
        help='Trigger filter kernel'
    )
    trigger_threshold = strax.Config(
        default=100.0, type=float, help='Trigger threshold in filtered trace units'
    )
    deactivation_threshold_coefficient = strax.Config(
        default=1.0, type=float,
        help='Coefficient that the trigger threshold is multiplied by to produce the deactivation threshold.'
             'A hit is defined by the interval from where the waveform crosses the activation threshold,'
             'to where it crosses the deactivation threshold in the opposite direction'
    )
    trigger_holdoff = strax.Config(
        default=0, type=int,
        help='Time in samples after each trigger when triggering is not allowed'
    )

    def setup(self):
        self.kernel = self.filter_kernel

    def select_records(self, raw_records):
        """
        A function that can be overridden by a child plugins to apply triggering only to the records selected by this
        methods. If not overridden, the triggering is applied to all the records.

        :param raw_records: input records
        :return: records selected for the triggering. All the records if the method is not overridden.
        """
        return raw_records

    def compute(self, raw_records):
        """
        Takes raw records, applies selection using the select_records method, convolves with the filter kernel specified
        in the config, applied double-threshold triggering.

        :param raw_records: structured array of input records
        :return: structured array of triggers
        """
        selected_records = self.select_records(raw_records)
        # todo: speed it up! Writing a vectorized convolve function that convolves a 2d array with a 1d kernel
        #  (note that scipy.signal.convolve2d is not what you want, it applied convolution to both axes).
        #  I think, the easiest way to do it is to use scipy or np FFT module, which is properly vectorized. One would
        #  need to correctly zero-pad the kernel, then apply FFT to the signals and the kernel, multiply one by another,
        #  apply iFFT, to the result, and remove the edges as it is done in the 'valid' mode in the signal.convolve.
        #  Or maybe this problem is already solved in np or scipy and I just didn't find the function I needed.
        filtered_records = np.array(
            [signal.convolve(raw_record, self.kernel, mode='valid') for raw_record in selected_records['data']])
        # threshold triggering function is in helix.utils as a common use function
        hits_in_records = hx.threshold_trigger_2d(filtered_records, self.trigger_threshold,
                                                  self.deactivation_threshold_coefficient,
                                                  trigger_holdoff=self.trigger_holdoff)

        hit_count = np.sum([len(hits[0]) for hits in hits_in_records])
        results = np.empty(hit_count, dtype=self.dtype)
        results = _fill_results(results, selected_records, hits_in_records)

        # since we passed only the 'valid' convolution to the threshold trigger function,
        # the times and indices must be shifted by half the kernel to correspond to the locations in the non-convolved
        # raw records
        shift = len(self.kernel)//2
        results['time'] += shift * selected_records['dt'][0]
        results['start_loc'] += shift
        results['loc'] += shift

        # todo: first len(kernel)//2 samples and last (len(kernel)-1)//2 samples can't have any triggers in them as they
        #  are not included in the convolution result with mode='valid'. Such record edges should be considered dead
        #  time. We need some way of calculating all the dead time, e.g. returning another data_type
        #  'dead_time_intervals', containing only the mandatory strax.time_fields fields.
        #  Another way of dealing with it (a better way, but more difficult and probably more computationally intensive)
        #  is to convert this plugin to a strax.OverlapWindowPlugin, which would read the neighboring records with each
        #  record. This would allow to add a piece of the previous record and a piece of the next record to each record,
        #  and calculate the full convolution.
        #  Another option is to see what they do in straxen. I think they determine how to split the data into records
        #  in some smart way such that there are no events on record edges. This probably means that records have
        #  different length.

        # sorting by time (and the by channels if time is the same)
        return np.sort(results, order=['time', 'channel'])


@numba.njit(cache=True, nogil=True)
def _fill_results(results, records, hits_in_records):
    """
    JIT'ted function to copy trigger information to the structured array of trigger dtype

    :param results: empty structured array of helix.triggers_dtype to be filled out. The length should be equal to the
    total number of hits in hits_in_records
    :param records: structured array of raw_records
    :param hits_in_records: list of tuples (max_locations, start_locations, lengths, amplitudes, deactivation_crossed) -
    output of the helix.threshold_trigger_2d function (see helix/utils.py). Each tuple element is an array of triggers'
    information corresponding to all triggers in one records. The list length should be equal to len(records).
    :return: filled out structured array of triggers_dtype
    """
    k = 0
    for i, record in enumerate(records):
        # unpacking hits
        max_locations, start_locations, lengths, amplitudes, deactivation_crossed = hits_in_records[i]
        for j in range(len(max_locations)):
            results['time'][k] = record['time'] + start_locations[j]*record['dt']
            results['length'][k] = lengths[j]
            results['dt'][k] = record['dt']
            results['channel'][k] = record['channel']
            results['channel_type'][k] = record['channel_type']
            results['block_id'][k] = record['block_id']
            results['start_loc'][k] = start_locations[j]
            results['loc'][k] = max_locations[j]
            results['amplitude'][k] = amplitudes[j]
            results['deactivation_crossed'][k] = deactivation_crossed[j]
            k += 1

    return results


@export
class UVTriggers(Triggers):
    """
    Plugin for ultraviolet (UV)-signal triggering. Applies a Gaussian Derivative Filter (GDF) of configurable width to
    raw records, then applies double-threshold triggering. Inherits the logic of the base helix.Triggers plugin.
    """
    child_plugin = True  # this lets strax know that the configs and everything else should be inherited from Triggers
    provides = 'uv_triggers'
    # we need to have different data_kinds for UV and QP triggers, because they correspond to different time intervals
    data_kind = 'uv_triggers'
    __version__ = '0.0.0'

    uv_gdf_kernel_sigma = strax.Config(
        default=50, type=int,
        help='Gaussian sigma (in samples) of the Gaussian derivative filter kernel of the UV signal trigger'
    )

    def setup(self):
        """
        Uses the config to build the Gaussian Derivative Filter kernel used for the convolutions in the helix.Triggers
        base plugin.
        """
        super().setup()
        self.kernel = hx.get_gdf_kernel(self.uv_gdf_kernel_sigma)


@export
class QPTriggers(Triggers):
    """
    Plugin for quasiparticle (QP)-signal triggering in vacuum channels. Applies a Gaussian Derivative Filter (GDF) of
    configurable width to raw records, then applies double-threshold triggering. Inherits the logic of the base
    helix.Triggers plugin.
    """
    child_plugin = True
    provides = 'qp_triggers'
    data_kind = 'qp_triggers'
    __version__ = '0.0.0'

    qp_gdf_kernel_sigma = strax.Config(
        default=500, type=int,
        help='Gaussian sigma (in samples) of the Gaussian derivative filter kernel of the QP signal trigger'
    )

    def select_records(self, raw_records):
        """
        Selects records corresponding to the vacuum channels.

        :param raw_records: structured array of raw records
        :return: structured array of raw records in vacuum channels
        """
        return raw_records[raw_records['channel_type'] == hx.ChannelType.VACUUM]

    def setup(self):
        """
        Uses the config to build the Gaussian Derivative Filter kernel used for the convolutions in the helix.Triggers
        base plugin.
        """
        super().setup()
        self.kernel = hx.get_gdf_kernel(self.qp_gdf_kernel_sigma)
