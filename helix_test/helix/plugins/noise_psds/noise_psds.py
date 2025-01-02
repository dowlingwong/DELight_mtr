import strax
import numpy as np
from helix import units
import helix as hx

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
class NoisePSDs(strax.Plugin):
    """
    Plugin to calculate noise Power Spectrum and Cross-Spectrum Densities (PSDs and CSDs) from the noise events
    """
    depends_on = 'noise_events'
    provides = 'noise_psds'
    data_kind = 'noise_psds'
    parallel = False  # this plugin keeps state (cached PSD) between the compute() calls, so it can't be parallel

    # during development, it is useful to set plugin version to None. Strax will then hash the code itself when
    # calculating the lineage hash. If the code changes, strax will know that the version changed. It allows to rerun
    # some tests multiple times, changing the plugin code in-between, without deleting produced test data
    # everytime. Strax will see that the code changed, so it will recalculate the test data. Otherwise, it would not
    # recalculate the data, unless you manually change the plugin version every time you slightly changed the code.
    # When the plugin is developed, don't forget to assign some version to it.
    __version__ = None

    channel_map = strax.Config(
        default=hx.DEFAULT_CHANNEL_MAP, type=dict,
        help='Dictionary mapping channel types (SUBMERGED, VACUUM) to the corresponding range of channel numbers'
    )
    of_length = strax.Config(
        default=hx.DEFAULT_OF_LENGTH, type=int,
        help='Trace interval length (in samples) for the Optimum Filter fits'
    )
    noise_psd_duration = strax.Config(
        default=hx.DEFAULT_NOISE_PSD_DURATION, type=int,
        help='Time intervals for which noise PSDs are calculated in ns'
    )

    summed_channel_types = [hx.ChannelType.SUBMERGED_SUMMED, hx.ChannelType.VACUUM_SUMMED, hx.ChannelType.SUMMED]

    def __init__(self):
        super().__init__()
        # we will cache two PSD instances before returning them (by instance I mean one entry of the structured array
        # of noise_psds_dtype): last finished PSD, and the current PSD, which might not be finished yet, meaning, strax
        # did not pass enough data yet to the plugin to get the required "noise_psd_duration"
        self.finished_cached_psd = None  # actually, this variable will store a 1-sized array of PSDs
        self.unfinished_cached_psd = None  # actually, this variable will store a 1-sized array of PSDs
        # shows the expected endtime of the unfinished PSD instance
        self.expected_psd_end = -1
        self.last_psd_end = None

    def infer_dtype(self):
        n_channels = len(hx.Channels(self.channel_map))
        return hx.get_psds_dtype(self.of_length, n_channels, len(self.summed_channel_types))

    def setup(self):
        self.channels = hx.Channels(self.channel_map)
        self.n_channels = len(self.channels)
        self.inferred_dtype = self.infer_dtype()

    # overloading Plugin's iter method to return the cached PSDs at the end
    def iter(self, iters, executor=None):
        """
        Overloading strax.Plugin.iter() method to return all the cached PSDs after strax has completed feeding data to
        the plugin. This is necessary, because when strax calls the compute method, passing more and more data to it,
        we never know whether there is more data to come. When there is no more data, we need to return the unfinished
        PSD. If the unfinished psd_dtype instance is shorter than the required noise_psd_duration, the unfinished PSD
        is merged with the last finished PSD. That's why we keep 2 PSDs in the plugin's state at all times.

        :param iters: see strax.Plugin.iter()
        :param executor: see strax.Plugin.iter()
        :return: see strax.Plugin.iter()
        """
        # doing what strax.Plugin.iter should do
        yield from super().iter(iters, executor=executor)

        if self.finished_cached_psd or self.unfinished_cached_psd:
            # when all the data was passed to the plugin, returning the last (cached) PSDs.
            # merge the cached PSDs if necessary
            yield self._merge_cached_psds()

    # note that we added start and end parameters to the compute method. Strax supports this feature. It inspects the
    # compute method signature, and if contains start and end fields, it assigns start and end times of the data chunk
    # to them. In other plugins we did not need chunk starts and ends, so we did not include them in the compute()
    # signature
    def compute(self, noise_events, start, end):
        """
        Computes noise PSDs from noise events. A chunk of noise events might correspond to a time interval shorter than
        what we require for one PSD instance. In that case, we store the unfinished PSD in the state of the class until
        we collect enough data (data covering the requested noise_psd_duration).

        :param noise_events: structured array of noise events
        :param start: start time of the noise_events chunk in unix timestamps in ns.
        :param end: end time of the noise_events chunk in unix timestamps in ns
        :return:
        """
        # if this is the first data chunk, expected_psd_end is negative (see init). The first PSD should cover the time
        # frame from of noise_psd_duration starting at the start of the first chunk (start)
        if self.expected_psd_end < 0:
            self.expected_psd_end = start + self.noise_psd_duration

        psds = []
        t = start
        is_last_psd_complete = False
        while t < end:
            # we want to keep the boundaries of each PSD at the block boundaries rather than at the noise events
            # boundaries, to make sure that each data block is described by one PSD (while one PSD can describe more
            # than one data blocks). (Reminder: data block corresponds to one record, but spans all the channels)

            # one chunk of noise events can (and usually does) contain multiple data blocks. It is possible, that the
            # expected boundary of the current PSD is within this chunk. In that case, we split the chunk into multiple
            # PSDs
            mask = (noise_events['block_time'] >= t) & (noise_events['block_time'] < self.expected_psd_end)
            is_last_psd_complete = False

            if mask.sum():
                # if the last event in the chunk belongs to the current PSD (if mask[-1]==True), the split time is equal
                # to the chunk end time. Otherwise, it is at the boundary of data blocks closest to the expected PSD end
                split_time = end if mask[-1]\
                    else noise_events['block_time'][np.argmin(np.diff(mask.astype(int))) + 1]
            else:
                split_time = noise_events['block_time'][np.argmax(noise_events['block_time'] >= t)]

            # calculate PSDs and CSDs for events to the left from the split_time
            psds.append(self._get_psd(noise_events[mask], t, split_time)[0])
            # advance t to the split_time
            t = split_time
            # if expected_psd_end is within the chunk, the last calculated PSD is complete.
            if self.expected_psd_end <= end:
                # calculate the expected end of the next PSD
                self.expected_psd_end = t + self.noise_psd_duration
                is_last_psd_complete = True

        # if we have an unfinished cached PSD, we need to combine it with the first (or the only) PSD calculated for the
        # events in the current chunk
        if self.unfinished_cached_psd:
            if len(psds) > 1 or is_last_psd_complete:
                # if we got a new complete PSD, we store in the finished_cached_psd.
                if self.finished_cached_psd:
                    # if we already had one stored their, we now store them both, and will return one of them in the end
                    # of this function.
                    # Note that we concatenate two PSDs, that is we get a list of two PSDs, rather than merging two PSDs
                    # into one.
                    # One of the two PSDs is a merge of the previously unfinished PSD and the freshly calculated PSD.
                    self.finished_cached_psd = np.concatenate([self.finished_cached_psd,
                                                               self._combine_psds([self.unfinished_cached_psd[0],
                                                                                   psds.pop(0)])])
                else:
                    # since we had an unfinished PSD, we merge it with the PSD calculated in this round
                    self.finished_cached_psd = self._combine_psds([self.unfinished_cached_psd[0], psds.pop(0)])
                # the unfinished PSD was merged with the new one, so we don't currently have an unfinished PSD
                self.unfinished_cached_psd = None
            else:
                # merge the unfinished PSD with the PSD calculated in this round, but it is still unfinished.
                self.unfinished_cached_psd = self._combine_psds([self.unfinished_cached_psd[0], psds.pop(0)])
                # after this, 'psds' is an empty list

        # dealing with the unfinished cached PSD, if we still have freshly calculated PSDs
        if psds:
            # if the first one doesn't contain any events, merge it with the next one
            if psds[0]['n_events'] == 0:
                psds[1] = self._combine_psds(psds[:2])
                psds = psds[1:]

            if is_last_psd_complete:
                # no unfinished PSDs
                self.unfinished_cached_psd = None
            else:
                # the last psd in the list is unfinished. storing it in self.unfinished_cached_psd
                self.unfinished_cached_psd = np.empty(1, dtype=self.inferred_dtype)
                self.unfinished_cached_psd[0] = psds.pop()

        # dealing with the finished cached PSD, if we still have freshly calculated PSDs in the list
        if psds:
            if self.finished_cached_psd:
                # concatenating all the finished psds
                self.finished_cached_psd = np.concatenate([self.finished_cached_psd, np.array(psds)])
            else:
                # or storing all the finished psds in self.finished_cached_psd
                self.finished_cached_psd = np.array(psds)

        if self.finished_cached_psd is not None and len(self.finished_cached_psd) > 1:
            # returning all the finished psds, except the last one. That one we will store until the next iter() call
            psds_to_return = self.finished_cached_psd[:-1]
            self.finished_cached_psd = self.finished_cached_psd[-1:]
            self.last_psd_end = psds_to_return[-1]['endtime']
            # see strax.Plugin._fix_output(): converts structured array into strax.Chunk.
            # usually, strax calls this function automatically when compute() returns an array instead of a Chunk.
            # Here we call it manually to manually set the chunk boundaries to correspond to the PSD boundaries, rather
            # than to the input noise_event chunk boundaries
            return self._fix_output(psds_to_return, psds_to_return[0]['time'], psds_to_return[-1]['endtime'])
        else:
            # we stored some PSDs, but there is nothing to return. Returning an empty chunk
            return self._get_empty_chunk(start)

    def _get_empty_chunk(self, time):
        """
        Returns an empty chunk with an empty noise_psd_dtype array

        :param time: time to assign to the chunk, to ensure that chunks are properly ordered.
        :return: empty strax.Chunk
        """
        # even when returning an empty chunk, strax wants it to be properly ordered. That's why we keep the
        # last_psd_end.
        if self.last_psd_end is None:
            self.last_psd_end = time
        return self._fix_output(np.empty(0, dtype=self.inferred_dtype),
                                self.last_psd_end, self.last_psd_end)

    def _get_psd(self, events, start, end):
        """
        Calculate PSDs and CSDs of provided events.

        :param events: Structured array of noise events to calculate the PSDs from
        :param start: start time of the PSD instance (unix timestamp in ns)
        :param end: end time of the PSD instance (unix timestamp in ns)
        :return: 1-sized structured array of noise_psd_dtype with the calculated PSDs and CSDs
        """
        # 1-sized output PSD array
        psd = np.zeros(1, dtype=self.inferred_dtype)
        # set the time and endtime
        psd['time'] = start
        psd['endtime'] = end
        # if no events, leave PSD and CSD data equal to 0
        if not len(events):
            return psd
        psd['n_events'] = len(events)
        psd['channels'] = events['channels'][0]
        sampling_frequency = 1 / (events['dt'][0] / units.s)  # Hz
        psd['frequencies'], psd['psds'] = hx.calculate_psd(events['channel_data'], sampling_frequency)
        _, psd['summed_channel_psds'] = hx.calculate_psd(events['data'], sampling_frequency)
        psd['summed_channel_types'] = events['summed_channel_types'][0]

        # calculating CSDs for each pair of channels
        # Going through the loop of channels (excluding the last one), let's call it channel A,
        # and in each loop iteration calculate CSDs for all (A, B) pairs, where B > A (by channel index)
        # (to avoid calculating CSDs for the same pair twice, first for (A, B), and then for (B, A)).
        for i_ch_a, ch in enumerate(self.channels.numbers[:-1]):
            ch_b_start = i_ch_a + 1  # index of channel B must be larger than the index of channel A
            ch_b_end = self.n_channels
            # this function takes a pair of channel indices and returns a CSD index
            # we call it with the first B and the last B to get a range of CSD indices for the current loop iteration
            csd_start = hx.get_csd_index(self.n_channels, (i_ch_a, ch_b_start))
            csd_end = hx.get_csd_index(self.n_channels, (i_ch_a, self.n_channels - 1))
            # filling out the csds field in the output array
            _, psd['csds'][0, csd_start: csd_end + 1] = hx.calculate_csd(events['channel_data'][:, i_ch_a:i_ch_a+1],
                                                                         events['channel_data'][:, ch_b_start:ch_b_end],
                                                                         sampling_frequency)
        return psd

    def _combine_psds(self, psds):
        """
        Combines PSD instances into one PSD with more events. So, if one PSD instance is calculated on 10 events,
        and another one on 20 events, this function combines the two PSDs to create one PSD instance calculated on 20
        events.
        :param psds: list of psds to combine. Elements of the list are numpy records (not to be confused with helix
        records) of noise_psd_dtype.
        :return: 1-sized structured array of noise_psd_dtype with a combined PSD
        """
        event_counts = [psd['n_events'] for psd in psds]
        n_events = np.sum(event_counts)
        if n_events == 0:
            raise ValueError('Provided PSDs in NoisePSDs._combine_psds do not have any events')
        result = np.empty(1, dtype=self.inferred_dtype)
        result[0] = np.copy(psds[np.argmax(event_counts)])
        r = result[0]
        r['n_events'] = n_events
        for key in ['psds', 'summed_channel_psds', 'csds']:
            r[key] = np.sum([psd['n_events'] * psd[key] for psd in psds], axis=0) / n_events
        r['time'] = min([psd['time'] for psd in psds])
        r['endtime'] = max([psd['endtime'] for psd in psds])
        return result

    def _merge_cached_psds(self):
        """
        Combines unfinished and finished cached PSDs, returns them as a strax.Chunk
        :return: strax.Chunk containing combined cached PSDs
        """
        if self.finished_cached_psd and self.unfinished_cached_psd:
            merged_psd = self._combine_psds([self.unfinished_cached_psd[0], self.finished_cached_psd[0]])
        elif self.finished_cached_psd:
            merged_psd = self.finished_cached_psd
        else:
            merged_psd = self.unfinished_cached_psd

        return self._fix_output(merged_psd, merged_psd[0]['time'], merged_psd[0]['endtime'])
