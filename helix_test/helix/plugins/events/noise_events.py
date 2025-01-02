import strax
import numpy as np
from helix import units
import helix as hx
import numba
from datetime import datetime

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
class NoiseEvents(strax.Plugin):
    """
    Plugin to find noise samples. Takes raw records and UV and QP triggers, and finds intervals in the records without
    any triggers in them.
    """
    depends_on = ('raw_records', 'qp_triggers', 'uv_triggers')
    provides = 'noise_events'
    # data_kind is different from the Events plugin, since noise_events and events do not correspond to the same times
    data_kind = 'noise_events'
    __version__ = '0.0.0'

    channel_map = strax.Config(
        default=hx.DEFAULT_CHANNEL_MAP, type=dict,
        help='Dictionary mapping channel types (SUBMERGED, VACUUM) to the corresponding range of channel numbers'
    )
    of_length = strax.Config(
        default=hx.DEFAULT_OF_LENGTH, type=int,
        help='Trace interval length (in samples) for the Optimum Filter fits'
    )
    pre_trigger_noise_trace_veto = strax.Config(
        default=hx.DEFAULT_PRE_TRIGGER_NOISE_TRACE_VETO, type=int,
        help='Minimum number of samples between the end of a noise trace and a trigger'
    )
    post_trigger_noise_trace_veto = strax.Config(
        default=hx.DEFAULT_POST_TRIGGER_NOISE_TRACE_VETO, type=int,
        help='Minimum number of samples between a trigger and the start of a noise trace'
    )
    n_noise_events_per_record = strax.Config(
        default=hx.DEFAULT_N_NOISE_EVENTS_PER_RECORD, type=int,
        help='Number of noise events per record'
    )
    allow_noise_events_overlaps = strax.Config(
        default=hx.DEFAULT_ALLOW_NOISE_EVENTS_OVERLAPS, type=bool,
        help='If true, noise traces can overlap with each other'
    )
    noise_events_random_seed = strax.Config(
        default=hx.DEFAULT_NOISE_EVENTS_RANDOM_SEED, type=int,
        help='Seed for random number generator for noise events. If None, random seed is used'
    )

    # order of summed channel types in the 'data' field of the output structured array.
    # no sums of triggered channels here, because there are no triggers in the noise events
    summed_channel_types = [hx.ChannelType.SUBMERGED_SUMMED, hx.ChannelType.VACUUM_SUMMED, hx.ChannelType.SUMMED]

    def infer_dtype(self):
        n_channels = len(hx.Channels(self.channel_map))
        # using the same dtype as for the Events plugin, but with a different number of summed channel types
        return hx.get_events_dtype(self.of_length, n_channels, len(self.summed_channel_types))

    def setup(self):
        self.channels = hx.Channels(self.channel_map)
        self.n_channels = len(self.channels)
        self.inferred_dtype = self.infer_dtype()
        self.rng = np.random.default_rng(self.noise_events_random_seed)
        # summed channel masks can be calculated on setup, since the masks are always the same
        # (unlike in the Events plugin)
        self.summed_channel_masks = [self.channels.types == hx.ChannelType.SUBMERGED,
                                     self.channels.types == hx.ChannelType.VACUUM,
                                     np.ones_like(self.channels.types, dtype=np.bool_)]

    def _find_noise_events(self, dts, locs, n):
        """
        Returns random locations for noise events by the given inter-trigger intervals.

        :param dts: array of interval length between adjacent triggers in time samples
        :param locs: array of trigger locations in time samples. len(locs) == len(dts)
        :param n: requested number of noise events
        :return: array of non-intersecting noise events' starts in time samples. The returned array can be shorter than
        n, if there is not enough space for n events.
        """
        max_n_traces_per_dt = (dts - self.pre_trigger_noise_trace_veto -
                               self.post_trigger_noise_trace_veto - 1) // self.of_length

        # sometimes there is not enough trigger-free samples to get n noise traces
        # getting as many as we can
        n_traces = min(n, max_n_traces_per_dt.sum())

        # choosing interval indices to get the traces from
        chosen_dt_indices = self.rng.choice(np.repeat(np.arange(len(dts)), max_n_traces_per_dt), n_traces,
                                         replace=False, shuffle=False)

        # counting unique indices to know how many events to draw from each interval
        indices, counts = np.unique(chosen_dt_indices, return_counts=True)

        results = []
        # for each inter-trigger interval selected for drawing random events from, we need m+1 random floats to
        # determine the offsets of the random events from one another and from the interval edges, where m is the number
        # of noise events to be drawn from that interval.
        randoms = self.rng.random(counts.sum() + len(counts))
        j = 0
        for i, m in zip(indices, counts):
            # getting some random floats
            shifts = randoms[j: j + m + 1]
            # number of samples in an interval that are not going to be in the noise events
            n_skipped_samples = (dts[i] - 1 - m * self.of_length -
                                 self.pre_trigger_noise_trace_veto - self.post_trigger_noise_trace_veto)
            # normalizing random floats to get random shifts between the noise events. Rounding to int
            shifts = np.rint(shifts / shifts.sum() * n_skipped_samples).astype(np.int32)
            # correcting the possible rounding error
            shifts[0] = shifts[0] + n_skipped_samples - shifts.sum()
            # collecting start locations of noise events
            results.append(locs[i] + self.post_trigger_noise_trace_veto + np.cumsum(shifts)[:-1]
                           + np.arange(m) * self.of_length)
            # skipping m+1 used floats
            j += m + 1

        return np.concatenate(results)

    def compute(self, raw_records, qp_triggers, uv_triggers):
        # TODO: make a loop to iterate over separate block_ids. We wrote FakeDaqRawRecords plugin in such a way that one
        #  chunk of raw_records corresponds to one block_id. But in principle one could create another plugin for
        #  raw records in which one chunk contains more than one block_id. In that case we must iterate over different
        #  block_ids to properly locate the corresponding records.
        triggers = np.sort(np.concatenate((qp_triggers, uv_triggers)), order='loc')
        # calculating the time intervals between adjacent triggers (of any type, QP or UV), between 0 and the first
        # trigger, and between the last trigger and the record end
        trigger_dts = np.diff(triggers['loc'], prepend=0, append=raw_records['length'][0])
        trigger_locs = np.concatenate(([0,], triggers['loc']))  # prepending 0 to the trigger locations

        n_events = self.n_noise_events_per_record

        if self.allow_noise_events_overlaps:
            # selecting intervals ("dts") longer than the OF length and the sum of pre and post trigger vetoes
            mask = (trigger_dts > self.of_length + self.post_trigger_noise_trace_veto +
                    self.pre_trigger_noise_trace_veto)

            # building an array of all allowed event locations
            # e.g., if there is a trigger at loc=100, and the next one is at loc=300, and the pre and post trigger
            # vetoes are 10 samples, then any allowed locations are between 111 and 290
            allowed_locs = np.concatenate([np.arange(loc + self.post_trigger_noise_trace_veto + 1,
                                                     loc + dt - self.pre_trigger_noise_trace_veto - self.of_length - 1)
                                           for loc, dt in zip(trigger_locs[mask], trigger_dts[mask])])

            # if there are fewer allowed locs than the requested number of events, we can only return as many events as
            # there are allowed locs. But this is really an edge case, it should never happen. Unless vetoes are too
            # long. In that case allowed_locs can be empty, so we won't return any noise events
            n = min(n_events, len(allowed_locs))
            # choosing random locations for noise events
            event_locs = self.rng.choice(allowed_locs, n, replace=False, shuffle=False)
        else:
            # noise events must not overlap
            # calculating how many noise events can fit into each inter-trigger interval ("dt")
            max_n_traces_per_dt = (trigger_dts - self.pre_trigger_noise_trace_veto
                                   - self.post_trigger_noise_trace_veto - 1) // self.of_length
            mask = max_n_traces_per_dt > 0
            event_locs = self._find_noise_events(trigger_dts[mask], trigger_locs[mask], n_events)

        events = np.empty(len(event_locs), dtype=self.inferred_dtype)

        # reusing the fill_events function used for the Events plugin to fill out structured array of noise events
        return hx.fill_events(events, raw_records, event_locs, self.of_length, self.channels, self.summed_channel_types,
                              self.summed_channel_masks)
