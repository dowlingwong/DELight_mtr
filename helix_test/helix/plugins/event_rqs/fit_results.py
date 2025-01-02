import strax
import numpy as np
import helix as hx
from helix import units
from datetime import datetime

import itertools
import time
from warnings import warn

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
class FitResults(strax.Plugin):
    """
    Plugin that applies Optimum Filter (OF) fits to events. Two-template OF is applied to the vacuum channels, with
    templates corresponding to UV and QP signals. Regular one UV template OF is applied to the submerged channels.
    """
    depends_on = ('noise_psds', 'events')
    provides = 'fit_results'
    # data_kind is events, because each fit_result correspond to exactly one event, so events and fit_results can be
    # merged into one structured array with of a combined dtype, containing all the fields of the fit_results_dtype and
    # events_dtype
    data_kind = 'events'
    # adding a new class property for long chunks that serve as a support
    # we want strax to iterate over events chunks, not worrying about noise_psd boundaries
    # we use this new property in do_compute and iter methods
    support_data_types = ('noise_psds',)

    __version__ = None

    # TODO: it would be nice if we could move shared configs, such as channel_map, of_length and many others to a
    #  separate place, so plugins could take them from there. What we have now is code duplication. If we want to change
    #  one of such shared configs, we would have to change it in all plugins that have such a config
    channel_map = strax.Config(
        default=hx.DEFAULT_CHANNEL_MAP, type=dict,
        help='Dictionary mapping channel types (SUBMERGED, VACUUM) to the corresponding range of channel numbers'
    )
    of_length = strax.Config(
        default=hx.DEFAULT_OF_LENGTH, type=int,
        help='Trace interval length (in samples) for the Optimum Filter fits'
    )
    templates_path = strax.Config(
        default='plugins/event_rqs/default_templates.npy', type=str,
        help='Path to the OF template file. The file should contain an np array with shape (2, L), '
             'with UV and QP templates of length L=of_length'
    )
    allowed_fit_shifts = strax.Config(
        default=hx.DEFAULT_ALLOWED_FIT_SHIFTS, type=tuple,
        help='Tuple of allowed left and right time shifts in the OF fits. In samples. Left shifts are negative.'
    )
    allowed_two_template_deltas = strax.Config(
        default=hx.DEFAULT_ALLOWED_TWO_TEMPLATE_DELTAS, type=tuple,
        help='Tuple of allowed relative time shifts of UV and QP templates. In samples'
    )
    fit_summed_triggered_channels = strax.Config(
        default=hx.DEFAULT_FIT_SUMMED_TRIGGERED_CHANNELS, type=tuple,
        help='If true, fit_results plugin will fit summed triggered channels. This is slow.'
    )

    def infer_dtype(self):
        # I am not using self.channels here, because I don't know whether infer_dtype() is called earlier than setup()
        # I am not sure whether it is safe to set self.channels in __init__(), because the config might change, so
        # the channel map might change as well
        channels = hx.Channels(self.channel_map)
        return hx.get_fit_results_dtype(channels.counts[hx.ChannelType.SUBMERGED],
                                        channels.counts[hx.ChannelType.VACUUM])

    def setup(self):
        # TODO: for the real data we might have different templates for each channel.
        templates = np.load(self.templates_path, allow_pickle=True)
        self.uv_template = templates[0]
        self.qp_template = templates[1]
        self.channels = hx.Channels(self.channel_map)
        self.n_channels = len(self.channels)
        self.inferred_dtype = self.infer_dtype()

    # overriding strax.Plugin iter method
    # It's a copy of strax.Plugin.iter() with additional special checks for support_data_types
    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results.

        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to. If None, will compute inside the
            plugin's thread.

        """
        pending_futures = []
        last_input_received = time.time()
        self.input_buffer = {d: None for d in self.depends_on}

        # Fetch chunks from all inputs. Whoever is the slowest becomes the
        # pacemaker
        pacemaker = None
        _end = float("inf")
        for d in self.depends_on:
            self._fetch_chunk(d, iters)
            if self.input_buffer[d] is None:
                raise ValueError(f"Cannot work with empty input buffer {self.input_buffer}")
            # don't allow support_data_types to be a pacemaker (actually, I am not sure, maybe it's ok)
            if self.input_buffer[d].end < _end and d not in self.support_data_types:
                pacemaker = d
                _end = self.input_buffer[d].end

        # To break out of nested loops:
        class IterDone(Exception):
            pass

        try:
            for chunk_i in itertools.count():
                # Online input support
                while not self.is_ready(chunk_i):
                    if self.source_finished():
                        # Chunk_i does not exist. We are done.
                        print("Source finished!")
                        raise IterDone()

                    if time.time() > last_input_received + self.input_timeout:
                        raise strax.InputTimeoutExceeded(
                            f"{self.__class__.__name__}:{id(self)} waited for "
                            f"more than {self.input_timeout} sec for arrival of "
                            f"input chunk {chunk_i}, and has given up."
                        )

                    print(
                        f"{self.__class__.__name__} with object id: {id(self)} "
                        f"waits for chunk {chunk_i}"
                    )
                    time.sleep(2)
                last_input_received = time.time()

                if pacemaker is None:
                    inputs_merged = dict()
                else:
                    if chunk_i != 0:
                        # Fetch the pacemaker, to figure out when this chunk ends
                        # (don't do it for chunk 0, for which we already fetched)
                        if not self._fetch_chunk(pacemaker, iters):
                            # Source exhausted. Cleanup will do final checks.
                            raise IterDone()
                    this_chunk_end = self.input_buffer[pacemaker].end

                    inputs = dict()
                    # Fetch other inputs (when needed)
                    for d in self.depends_on:
                        if d != pacemaker:
                            while (
                                    self.input_buffer[d] is None
                                    or self.input_buffer[d].end < this_chunk_end
                            ):
                                self._fetch_chunk(d, iters, check_end_not_before=this_chunk_end)
                        if d in self.support_data_types:
                            # always feed the entire buffer to the compute method.
                            # will check how to modify the buffer ~20 lines of code below
                            inputs[d] = self.input_buffer[d]
                        else:
                            inputs[d], self.input_buffer[d] = self.input_buffer[d].split(
                                t=this_chunk_end, allow_early_split=True
                            )

                    # If any of the inputs were trimmed due to early splits,
                    # trim the others too.
                    # Only perform this procedure for the data_types that are not support data types
                    main_data_types = [d for d in self.depends_on if d not in self.support_data_types] if \
                        self.support_data_types else self.depends_on
                    max_passes_left = 10
                    while max_passes_left > 0:
                        this_chunk_end = min([x.end for d, x in inputs.items() if d in main_data_types]
                                             + [this_chunk_end])
                        if len(set([x.end for d, x in inputs.items() if d in main_data_types])) <= 1:
                            break
                        for d in main_data_types:
                            inputs[d], back_to_buffer = inputs[d].split(
                                t=this_chunk_end, allow_early_split=True
                            )
                            self.input_buffer[d] = strax.Chunk.concatenate([back_to_buffer, self.input_buffer[d]])
                        max_passes_left -= 1
                    else:
                        raise RuntimeError(
                            f"{self} was unable to get time-consistent "
                            f"inputs after ten passess. Inputs: \n{inputs}\n"
                            f"Input buffer:\n{self.input_buffer}"
                        )

                    # decide what to keep in the buffer for support data types
                    if self.support_data_types:
                        # if inputs for main data types are empty, emptying out support data chunks
                        if min([inputs[d].end - inputs[d].start for d in main_data_types]) == 0:
                            for d in self.support_data_types:
                                # this is just an easy way to create an empty chunk
                                inputs[d], _ = self.input_buffer[d].split(t=0, allow_early_split=True)
                        else:
                            # otherwise, checking if we can discard some part of the support input_buffers
                            for d in self.support_data_types:
                                _, self.input_buffer[d] = self.input_buffer[d].split(
                                    t=this_chunk_end, allow_early_split=True
                                )

                    # Merge inputs of the same kind
                    inputs_merged = {
                        kind: strax.Chunk.merge([inputs[d] for d in deps_of_kind])
                        for kind, deps_of_kind in self.dependencies_by_kind().items()
                    }

                # Submit the computation
                if self.parallel and executor is not None:
                    new_future = executor.submit(self.do_compute, chunk_i=chunk_i, **inputs_merged)
                    pending_futures.append(new_future)
                    pending_futures = [f for f in pending_futures if not f.done()]
                    yield new_future
                else:
                    yield from self._iter_compute(chunk_i=chunk_i, **inputs_merged)

        except IterDone:
            # Check all sources are exhausted.
            # This is more than a check though -- it ensure the content of
            # all sources are requested all the way (including the final
            # Stopiteration), as required by lazy-mode processing requires
            for d in iters.keys():
                if self._fetch_chunk(d, iters):
                    raise RuntimeError(f"Plugin {d} terminated without fetching last {d}!")

            # This can happen especially in time range selections
            if hasattr(self.save_when, "values"):
                save_when = max([int(save_when) for save_when in self.save_when.values()])
            else:
                save_when = self.save_when
            if save_when > strax.SaveWhen.EXPLICIT:
                for d, buffer in self.input_buffer.items():
                    # Check the input buffer is empty
                    if buffer is not None and len(buffer):
                        raise RuntimeError(f"Plugin {d} terminated with leftover {d}: {buffer}")

        finally:
            self.cleanup(wait_for=pending_futures)

    # overriding Plugin.do_compute too, to make sure that support data types do not define chunk's start and end fields
    def do_compute(self, chunk_i=None, **kwargs):
        """Wrapper for the user-defined compute method.

        This is the 'job' that gets executed in different processes/threads during multiprocessing

        """
        for k, v in kwargs.items():
            if not isinstance(v, strax.Chunk):
                raise RuntimeError(
                    f"do_compute of {self.__class__.__name__} got a {type(v)} "
                    f"instead of a strax Chunk for {k}"
                )

        start, end = None, None
        dep_by_kind = self.dependencies_by_kind()
        for k, v in kwargs.items():
            if dep_by_kind[k][0] not in self.support_data_types:
                start = v.start
                end = v.end
                break

        kwargs = {k: v.data for k, v in kwargs.items()}
        if self.compute_takes_chunk_i:
            kwargs["chunk_i"] = chunk_i
        if self.compute_takes_start_end:
            kwargs["start"] = start
            kwargs["end"] = end
        result = self.compute(**kwargs)

        return self._fix_output(result, start, end)

    def compute(self, noise_psds, events):
        # TODO: I think this plugin is horrible if PSDs correspond to long time intervals. Strax loads all the events
        #  corresponding to one such PSD. This could take an enormous amount of memory. This is a deal-breaking problem.
        #  An easy solution (but not very disk-space efficient) is to change the NoisePSDs plugin, so it splits one PSD
        #  into copies of itself spanning smaller time intervals.

        # TODO: implement sliding OF (rather than a rolling one)
        #  currently, I am not taking advantage of the fact that events are slightly longer than the fit window.
        #  I simply cut out a piece of event equal to the fit window (i.e. the template length)
        l = -self.allowed_fit_shifts[0]
        r = -self.allowed_fit_shifts[1]

        results = np.zeros(len(events), dtype=self.inferred_dtype)
        if not len(events):
            return results

        # copying fields that are the same in both events and fit_results
        for key in ['time', 'length', 'dt', 'block_id', 'event_id', 'channels']:
            results[key] = events[key]

        sampling_frequency = 1 / (events[0]['dt'] / units.s)  # Hz

        for psd in noise_psds:
            # events belonging to the current PSD
            event_mask = (events['time'] >= psd['time']) & (events['time'] < psd['endtime'])

            # if no events belong to this PSD, go to the next one
            if event_mask.sum() == 0:
                continue

            # TODO: Keep the last used PSDs saved. Only update OF objects, if PSDs are different from the previous ones.

            # taking advantage of OptimumFilter vectorization: passing multiple PSDs to it, corresponding to different
            # channels
            submerged_mask = self.channels.numbers_to_types(psd['channels']) == hx.ChannelType.SUBMERGED
            submerged_channel_of = hx.OptimumFilter(self.uv_template, psd['psds'][submerged_mask],
                                                    sampling_frequency, self.allowed_fit_shifts)
            i = np.argmax(psd['summed_channel_types'] == hx.ChannelType.SUBMERGED_SUMMED)
            # OF for the sum of submerged channels
            submerged_sum_of = hx.OptimumFilter(self.uv_template, psd['summed_channel_psds'][i],
                                                sampling_frequency, self.allowed_fit_shifts)

            # same for vacuum channels
            vacuum_mask = self.channels.numbers_to_types(psd['channels']) == hx.ChannelType.VACUUM
            vacuum_channel_of = hx.TwoTemplatesOptimumFilter(self.uv_template, self.qp_template,
                                                             psd['psds'][vacuum_mask], sampling_frequency,
                                                             self.allowed_fit_shifts, self.allowed_two_template_deltas)
            i = np.argmax(psd['summed_channel_types'] == hx.ChannelType.VACUUM_SUMMED)
            vacuum_sum_of = hx.TwoTemplatesOptimumFilter(self.uv_template, self.qp_template,
                                                         psd['summed_channel_psds'][i], sampling_frequency,
                                                         self.allowed_fit_shifts, self.allowed_two_template_deltas)

            i = np.argmax(psd['summed_channel_types'] == hx.ChannelType.SUMMED)
            sum_of = hx.TwoTemplatesOptimumFilter(self.uv_template, self.qp_template,
                                                  psd['summed_channel_psds'][i], sampling_frequency,
                                                  self.allowed_fit_shifts, self.allowed_two_template_deltas)

            # Fits of all the cases where the OF can be precalculated
            # Submerged
            submerged_channel_results = submerged_channel_of.fit(
                events[event_mask]['channel_data'][:, submerged_mask, l:r])
            for j, key in enumerate(['submerged_channel_uv_amplitude',
                                     'submerged_channel_uv_offset',
                                     'submerged_channel_fit_chi2']):  # the key order corresponds to OF.fit output order
                results[key][event_mask] = submerged_channel_results[j]

            i = np.argmax(events['summed_channel_types'][0] == hx.ChannelType.SUBMERGED_SUMMED)
            submerged_sum_results = submerged_sum_of.fit(events[event_mask]['data'][:, i, l:r])
            results['submerged_sum_uv_amplitude'][event_mask] = submerged_sum_results[0]
            results['submerged_sum_uv_offset'][event_mask] = submerged_sum_results[1]
            results['submerged_sum_fit_chi2'][event_mask] = submerged_sum_results[2]

            # Vacuum
            vacuum_channel_results = vacuum_channel_of.fit(events[event_mask]['channel_data'][:, vacuum_mask, l:r])
            for j, key in enumerate(['vacuum_channel_uv_amplitude',
                                     'vacuum_channel_qp_amplitude',
                                     'vacuum_channel_uv_offset',
                                     'vacuum_channel_qp_offset',
                                     'vacuum_channel_fit_chi2']):  # the key order = TwoTemplOF.fit output order
                results[key][event_mask] = vacuum_channel_results[j]

            i = np.argmax(events['summed_channel_types'][0] == hx.ChannelType.VACUUM_SUMMED)
            vacuum_sum_results = vacuum_sum_of.fit(events[event_mask]['data'][:, i, l:r])
            results['vacuum_sum_uv_amplitude'][event_mask] = vacuum_sum_results[0]
            results['vacuum_sum_qp_amplitude'][event_mask] = vacuum_sum_results[1]
            results['vacuum_sum_uv_offset'][event_mask] = vacuum_sum_results[2]
            results['vacuum_sum_qp_offset'][event_mask] = vacuum_sum_results[3]
            results['vacuum_sum_fit_chi2'][event_mask] = vacuum_sum_results[4]

            # Sum of all channels
            i = np.argmax(events['summed_channel_types'][0] == hx.ChannelType.SUMMED)
            sum_results = sum_of.fit(events[event_mask]['data'][:, i, l:r])
            results['sum_uv_amplitude'][event_mask] = sum_results[0]
            results['sum_qp_amplitude'][event_mask] = sum_results[1]
            results['sum_uv_offset'][event_mask] = sum_results[2]
            results['sum_qp_offset'][event_mask] = sum_results[3]
            results['sum_fit_chi2'][event_mask] = sum_results[4]

            # Triggered channels
            if self.fit_summed_triggered_channels:
                i_submerged_triggered_mask = np.argmax(events['summed_channel_types'][0] ==
                                                       hx.ChannelType.TRIGGERED_SUBMERGED_SUMMED)
                i_vacuum_triggered_mask = np.argmax(events['summed_channel_types'][0] ==
                                                    hx.ChannelType.TRIGGERED_VACUUM_SUMMED)
                i_triggered_mask = np.argmax(events['summed_channel_types'][0] ==
                                             hx.ChannelType.TRIGGERED_SUMMED)
                results['submerged_triggered_channel_masks'][event_mask] \
                    = events[event_mask]['summed_channel_masks'][:, i_submerged_triggered_mask]
                results['vacuum_triggered_channel_masks'][event_mask] \
                    = events[event_mask]['summed_channel_masks'][:, i_vacuum_triggered_mask]
                results['triggered_channel_masks'][event_mask] \
                    = events[event_mask]['summed_channel_masks'][:, i_triggered_mask]

                # SLOW: looping over each event. Recalculating PSD and OF for each event.
                # Fitting one trace at a time.
                for i_event, event in zip(np.arange(len(events))[event_mask], events[event_mask]):
                    # triggered submerged
                    if np.any(event['summed_channel_masks'][i_submerged_triggered_mask]):
                        triggered_psd = hx.calculate_psd_of_sum(psd['psds'], psd['csds'],
                                                                event['summed_channel_masks'][
                                                                    i_submerged_triggered_mask])
                        of = hx.OptimumFilter(self.uv_template, triggered_psd, sampling_frequency,
                                              self.allowed_fit_shifts)
                        fit_results = of.fit(event['data'][i_submerged_triggered_mask, l:r])
                        for j, key in enumerate(['submerged_triggered_uv_amplitude',
                                                 'submerged_triggered_uv_offset',
                                                 'submerged_triggered_fit_chi2']):
                            results[key][i_event] = fit_results[j]

                    # triggered vacuum
                    if np.any(event['summed_channel_masks'][i_vacuum_triggered_mask]):
                        triggered_psd = hx.calculate_psd_of_sum(psd['psds'], psd['csds'],
                                                                event['summed_channel_masks'][i_vacuum_triggered_mask])
                        of = hx.TwoTemplatesOptimumFilter(self.uv_template, self.qp_template,
                                                          triggered_psd, sampling_frequency,
                                                          self.allowed_fit_shifts, self.allowed_two_template_deltas)
                        fit_results = of.fit(event['data'][i_vacuum_triggered_mask, l:r])
                        for j, key in enumerate(['vacuum_triggered_uv_amplitude',
                                                 'vacuum_triggered_qp_amplitude',
                                                 'vacuum_triggered_uv_offset',
                                                 'vacuum_triggered_qp_offset',
                                                 'vacuum_triggered_fit_chi2']):
                            results[key][i_event] = fit_results[j]

                    # all triggered
                    triggered_psd = hx.calculate_psd_of_sum(psd['psds'], psd['csds'],
                                                            event['summed_channel_masks'][i_triggered_mask])
                    of = hx.TwoTemplatesOptimumFilter(self.uv_template, self.qp_template,
                                                      triggered_psd, sampling_frequency,
                                                      self.allowed_fit_shifts, self.allowed_two_template_deltas)
                    fit_results = of.fit(event['data'][i_triggered_mask, l:r])
                    for j, key in enumerate(['triggered_uv_amplitude',
                                             'triggered_qp_amplitude',
                                             'triggered_uv_offset',
                                             'triggered_qp_offset',
                                             'triggered_fit_chi2']):
                        results[key][i_event] = fit_results[j]

        return results
