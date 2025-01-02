import strax
import numpy as np
import helix as hx
import numba
from datetime import datetime

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
class Events(strax.Plugin):
    """
    Event-building plugin. Groups QP and UV triggers in different channels into one event if they are within a
    configurable time window from one another. Copies pieces of raw records around each such event for further
    processing
    """
    # when a plugin depends on multiple inputs, they are provided in the depends_on argument in a tuple:
    depends_on = ('raw_records', 'qp_triggers', 'uv_triggers')
    provides = 'events'
    data_kind = 'events'
    __version__ = None

    template_length = strax.Config(
        default=hx.DEFAULT_TEMPLATE_LENGTH, type=int,
        help='OF template length in samples'
    )
    template_prepulse_length = strax.Config(
        default=hx.DEFAULT_PREPULSE_LENGTH, type=int,
        help='Number of samples before the pulse in the OF template'
    )
    channel_map = strax.Config(
        default=hx.DEFAULT_CHANNEL_MAP, type=dict,
        help='Dictionary mapping channel types (SUBMERGED, VACUUM) to the corresponding range of channel numbers'
    )
    allowed_fit_shifts = strax.Config(
        default=hx.DEFAULT_ALLOWED_FIT_SHIFTS, type=tuple,
        help='Tuple of allowed left and right time shifts in the OF fits. In samples. Left shifts are negative'
    )

    # defining the order of summed channel types in the 'data' field of the output structured array
    summed_channel_types = [hx.ChannelType.SUBMERGED_SUMMED, hx.ChannelType.VACUUM_SUMMED, hx.ChannelType.SUMMED,
                            hx.ChannelType.TRIGGERED_SUBMERGED_SUMMED, hx.ChannelType.TRIGGERED_VACUUM_SUMMED,
                            hx.ChannelType.TRIGGERED_SUMMED]

    def infer_dtype(self):
        n_channels = len(hx.Channels(self.channel_map))
        return hx.get_events_dtype(self.template_length + self.allowed_fit_shifts[1] - self.allowed_fit_shifts[0],
                                   n_channels, len(self.summed_channel_types))

    def setup(self):
        self.channels = hx.Channels(self.channel_map)
        self.n_channels = len(self.channels)
        self.inferred_dtype = self.infer_dtype()
        self.template_postpulse_length = self.template_length - self.template_prepulse_length

    def compute(self, raw_records, qp_triggers, uv_triggers):
        """
        Groups the triggers into events. Cuts pieces of raw_records around each event. Calculates the sums of channels.
        Returns a structured array of events_dtype
        :param raw_records: structured array of raw records
        :param qp_triggers: structured array of QP triggers
        :param uv_triggers: structured array of UV triggers
        :return: structured array of events
        """
        # TODO: make a loop to iterate over separate block_ids. We wrote FakeDaqRawRecords plugin in such a way that one
        #  chunk of raw_records corresponds to one block_id. But in principle one could create another plugin for
        #  raw records in which one chunk contains more than one block_id. In that case we must iterate over different
        #  block_ids to properly locate the corresponding records.
        input_triggers = [qp_triggers, uv_triggers]
        record_length = raw_records['length'][0]

        # TODO: add the edges to the dead time somehow. Or make a strax.OverlapWindowPlugin and include the edges. Or
        #  define the records in a way that there are no events on their edges.

        # skipping triggers on record edges, since there is not enough space to build an event around them
        trigger_masks = [(triggers['loc'] >= self.template_prepulse_length - self.allowed_fit_shifts[0]) &
                         (triggers['loc'] <= record_length - self.template_postpulse_length -
                          self.allowed_fit_shifts[1]) for triggers in input_triggers]
        # concatenate UV and QP triggers together, excluding triggers on record edges
        trigger_locs = np.concatenate([triggers['loc'][mask]
                                       for triggers, mask in zip(input_triggers, trigger_masks)])
        # OF will be able to shift by -allowed_fit_shifts[0] samples to the left from event location, and by
        # allowed_fit_shifts[1] samples to the right. This means that event locations must be separated by at least
        # allowed_fit_shifts[1] - allowed_fit_shifts[0] to make sure that OF does not fit the same pulse more than once
        # (by reaching the same pulse while processing another event). Note that allowed_fit_shifts[0] is negative.
        event_locs = find_events(np.sort(trigger_locs), self.allowed_fit_shifts[1] - self.allowed_fit_shifts[0])

        # shifting the event start to the left to make room for left OF shifts and for the prepulse interval in the OF
        # template
        start_locs = event_locs + self.allowed_fit_shifts[0] - self.template_prepulse_length
        # the event length is equal to the OF fit window plus the allowed left and right OF shifts
        length = self.allowed_fit_shifts[1] - self.allowed_fit_shifts[0] + self.template_length

        events = np.empty(len(event_locs), dtype=self.inferred_dtype)

        summed_channel_masks = np.zeros((len(event_locs), len(self.summed_channel_types), self.n_channels),
                                        dtype=np.bool_)
        # TODO: see if you can speed up this loop and the fill_events function. If it is necessary. That is, if you find
        #  that it is a bottleneck in the processing pipeline. Note that currently it's a pure python loop over all the
        #  events in the record. Even two loops (the second one is in the fill_events function). So it is probably slow.
        for i, event_loc in enumerate(event_locs):
            # identifying which channel has triggers in an event
            triggered_channels = get_channels_with_triggers((qp_triggers, uv_triggers),
                                                            event_loc + self.allowed_fit_shifts[0],
                                                            event_loc + self.allowed_fit_shifts[1])

            channel_masks = [self.channels.types == hx.ChannelType.SUBMERGED,
                             self.channels.types == hx.ChannelType.VACUUM,
                             np.ones_like(self.channels.types, dtype=np.bool_)]

            # summed channel masks for each event has 6 elements, corresponding to the 6 summed_channel_types:
            # ChannelType.SUBMERGED_SUMMED - sum of all submerged channels
            # ChannelType.VACUUM_SUMMED - sum of all vacuum channels
            # ChannelType.SUMMED - sum of all channels
            # ChannelType.TRIGGERED_SUBMERGED_SUMMED - sum of the submerged channels that have a trigger in this event
            # ChannelType.TRIGGERED_VACUUM_SUMMED - sum of the vacuum channels that have a trigger in this event
            # ChannelType.TRIGGERED_SUMMED - sum of submerged and vacuum channels that have a trigger in this event
            summed_channel_masks[i] = channel_masks + [mask & np.isin(self.channels.numbers, triggered_channels)
                                                       for mask in channel_masks]

        return fill_events(events, raw_records, start_locs, length, self.channels, self.summed_channel_types,
                           summed_channel_masks)


def get_channels_with_triggers(trigger_groups, start_loc, end_index):
    masks = [(triggers['loc'] >= start_loc) & (triggers['loc'] < end_index)
             for triggers in trigger_groups]
    return np.unique(np.concatenate([triggers['channel'][mask] for triggers, mask in zip(trigger_groups, masks)]))


@export
@numba.njit(cache=True, nogil=True)
def find_events(sorted_trigger_locs, event_holdoff):
    """
    Assigns an event to each provided trigger, skipping the triggers that are within the event_holdoff time after
    the previous event. The function is JIT-accelerated.

    :param sorted_trigger_locs: trigger locations in samples
    :param event_holdoff: time (in samples) after each event in which other events are not allowed
    :return:
    """
    event_locs = []
    last_event_time = -1
    for t in sorted_trigger_locs:
        if t < last_event_time + event_holdoff:
            continue
        else:
            last_event_time = t
            event_locs.append(t)
    return np.array(event_locs)


@export
def fill_events(events, raw_records, start_locs, length, channels, summed_channel_types, summed_channel_masks):
    """
    Fills out the provided empty structured array of events_dtype with the record data and event information.

    :param events: empty structured array of events_dtype to be filled out
    :param raw_records: structured array of raw records to take the ADC data from, corresponding to one data block
    :param start_locs: array of event locations in the records (in time samples)
    :param length: length of each event in time samples
    :param channels: object of helix.Channels class describing the channels of the records
    :param summed_channel_types: list of types of summed channels for the 'data' field in the events_dtype
    :param summed_channel_masks: 2d array of booleans of shape (len(start_loc), len(summed_channel_types)) defining
    which channels to sum in each event and each summed channel type
    :return: filled out structured array of events
    """
    # TODO: multiply ADC data by channel gains. Add channel gains to the config.
    events['length'] = length
    events['dt'] = raw_records['dt'][0]
    events['block_id'] = raw_records['block_id'][0]
    events['block_time'] = raw_records['time'][0]
    events['start_loc'] = start_locs
    events['time'] = raw_records['time'][0] + start_locs.astype(np.int64) * raw_records['dt'][0]
    events['event_id'] = np.arange(len(start_locs))
    events['channels'] = channels.numbers
    events['summed_channel_types'] = summed_channel_types
    events['summed_channel_masks'] = summed_channel_masks

    # TODO: see if this loop is the bottleneck in the processing pipeline. If it is the case, try to accelerate this
    #  function with @numba.njit
    for event_id, loc in enumerate(start_locs):

        events['channel_data'][event_id] \
            = np.sort(raw_records, order='channel')['data'][channels.order, loc: loc + length]

        events['data'][event_id] = [np.sum(events['channel_data'][event_id][mask], axis=0)
                                    for mask in events['summed_channel_masks'][event_id]]

    return events
