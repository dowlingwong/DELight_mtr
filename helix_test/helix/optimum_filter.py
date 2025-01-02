import numpy as np
import scipy
import strax
import helix as hx
from warnings import warn

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()

UNSPECIFIED = -1


# Optimization ideas:

# 1. The _update_state() function can be split into update_template and update_psd, to save time in cases
# when we only want to change the PSD, keeping the template the same. It would save us a couple FFTs

# 2. We are dealing with real-valued traces. We can use folded PSDs and rfft/irfft.
# This would make all the FFTs more than 2 times faster. Special care should be taken with the PSD edges.

@export
class OptimumFilter:
    """
    Class to perform one-template Optimum Filter fits. Inspired by QETpy
    If noise_psd and template are 1-dimensional, traces can be any-dimensional.
    If noise_psd or template is 2-dimensional (i.e., N PSDs or N templates, or both N PSDs and N templates are
    provided), traces can be:
        1-dimensional: one trace, fitted with N different PSDs and/or templates. Outputs have shapes (N,)
        2-dimensional: must have shape (N, L), where L is the trace length. i-th trace is fitted with i-th PSD and/or
            template. The outputs have shapes (N,)
        3-dimensional: must have shape (M, N, L), where M is arbitrary. The outputs have shapes (M, N).
    """

    def __init__(self, template, noise_psd, sampling_frequency, allowed_shifts=None):
        """
        Makes an OF object to perform one-template Optimum Filter fits. Inspired by QETpy.
        If noise_psd and template are 1-dimensional, traces can be any-dimensional.
        If noise_psd or template is 2-dimensional (i.e., N PSDs or N templates, or both N PSDs and N templates are
        provided), traces can be:
            1-dimensional: one trace, fitted with N different PSDs and/or templates. Outputs have shapes (N,)
            2-dimensional: must have shape (N, L), where L is the trace length. i-th trace is fitted with i-th PSD
                and/or template. The outputs have shapes (N,)
            3-dimensional: must have shape (M, N, L), where M is arbitrary. The outputs have shapes (M, N).

        :param template: 1- or 2-dimensional array with either one template or N templates of length L
        :param noise_psd: 1- or 2-dimensional arrays of folded noise PSDs. If both template and noise_psd are
        2-dimensional, their first dimension must match.
        :param sampling_frequency: sampling frequency in Hz
        :param allowed_shifts: a tuple of allowed range of template rolls in time samples
        """
        self._template = template
        self._noise_psd = noise_psd
        self._allowed_shifts = allowed_shifts
        self._check_shapes()
        self._sampling_frequency = sampling_frequency
        self._update_state()

    def set_template(self, template):
        """
        Update the fit template
        :param template: 1- or 2-dimensional array with either one template or N templates of length L
        """
        self._template = template
        self._check_shapes()
        self._update_state()

    def set_noise_psd(self, noise_psd):
        """
        Update the noise PSD
        :param noise_psd: 1- or 2-dimensional arrays of folded noise PSDs. If both template and noise_psd are
        2-dimensional, their first dimension must match.
        """
        self._noise_psd = noise_psd
        self._check_shapes()
        self._update_state()

    def set_allowed_shifts(self, allowed_shifts):
        """
        Updated allowed range of template rolls
        :param allowed_shifts: a tuple of allowed range of template rolls in time samples
        """
        self._allowed_shifts = allowed_shifts
        self._check_shapes()

    def _update_state(self):
        """
        A function that recalculates the state of the OF object (i.e. the filter kernel and some other things)
        """
        self._length = self._template.shape[-1]

        self._noise_psd_unfolded = hx.unfold_psd(self._noise_psd)

        if np.any(np.sum(self._noise_psd_unfolded[..., 1:], axis=-1) == 0):
            warn('A zero PSD was passed to the OF. Disabling noise deweighting.')
            if len(self._noise_psd_unfolded.shape) > 1:
                self._noise_psd_unfolded[np.sum(self._noise_psd_unfolded[..., 1:], axis=-1) == 0] = (
                    np.ones(self._noise_psd_unfolded.shape[-1]))
            else:
                self._noise_psd_unfolded = np.ones(self._noise_psd_unfolded.shape[-1])

        elif np.any(self._noise_psd_unfolded[..., 1:] == 0):
            warn('One of the PSD values in the OF is 0. It will cause a division by 0 error')

        self._noise_psd_unfolded[..., 0] = np.inf

        self._template_fft = scipy.fft.fft(self._template) / self._sampling_frequency

        self._kernel_fft = self._template_fft.conjugate() / self._noise_psd_unfolded
        self._kernel_normalization = np.real(
            np.einsum('...j, ...j -> ...', self._kernel_fft,
                      self._template_fft)) * self._sampling_frequency / self._length

        self._resolution = 1 / np.sqrt(self._kernel_normalization)

    def _check_traces_shape(self, traces):
        """
        Checks whether the traces shape correspond to the OF templates and PSDs
        :param traces: array of traces to fit
        """
        if traces.shape[-1] != self._template.shape[-1]:
            raise ValueError(
                f'In the OF, traces length {(traces.shape[-1])} must be equal to the template length '
                f'({self._template.shape[-1]}).')

        if len(self._kernel_fft.shape) > 1 and len(traces.shape) > 1 and self._kernel_fft.shape[0] != traces.shape[-2]:
            raise ValueError(
                f'In OF with more than one PSD or template, the -2 axis of traces should be equal to the number of '
                f'PSDs or templates. Number of PSDs is {self._noise_psd_unfolded.shape[0]}. Number of templates is '
                f'{self._template.shape[0]}. The provided traces shape is {traces.shape}')

    def _check_shapes(self):
        """
        Checks whether the shapes of OF templates and PSDs match
        """
        if len(self._noise_psd.shape) > 2:
            raise NotImplementedError(
                f'PSDs with more than 2 dimensions are not supported. Your PSD shape is {self._noise_psd.shape}.')
        if len(self._template.shape) > 2:
            raise NotImplementedError(
                f'Templates with more than 2 dimensions are not supported. '
                f'Your template shape is {self._template.shape}.')
        if len(self._noise_psd.shape) > 1 and len(self._template.shape) > 1 and self._noise_psd.shape[0] != \
                self._template.shape[0]:
            raise NotImplementedError(
                f'If templates and PSDs are 2-dimensinonal, their lengths along the 0-axis must match. '
                f'You provided {len(self._template)} templates and {len(self._noise_psd)} PSDs.')

    @property
    def resolution(self):
        """
        Expected energy resolution, based on the template and the noise PSD. Has the length of N, if either N templates
        or N PSDs are provided
        """
        return self._resolution

    def fit_with_no_shift(self, traces):
        """
        Performs Optimum Filter fit to the traces with no template shifts.
        :param traces: array of traces to fit. If OF has N templates and/or N PSDs, traces can be:
            1-dimensional: one trace, fitted with N different PSDs and/or templates. Outputs have shapes (N,).
            2-dimensional: must have shape (N, L), where L is the trace length. i-th trace is fitted with i-th PSD
                and/or template. The outputs have shapes (N,)
            3-dimensional: must have shape (M, N, L), where M is arbitrary. The outputs have shapes (M, N).
        :return: tuple of fit results: (amplitudes, reduced_chi_squared_values)
        """
        self._check_traces_shape(traces)

        trace_fft = scipy.fft.fft(traces, axis=-1) / self._sampling_frequency  # V
        trace_filtered = self._kernel_fft * trace_fft / self._kernel_normalization[..., np.newaxis]  # Phi * V
        chisq0 = np.real(
            np.einsum('...j, ...j -> ...',
                      trace_fft.conjugate() / self._noise_psd_unfolded,
                      trace_fft) * self._sampling_frequency / self._length
        )

        amp0 = np.real(np.sum(
            trace_filtered, axis=-1
        )) * self._sampling_frequency / self._length

        # total chisq
        # self._length-2 is the number of degrees of freedom. Is this actually correct? Shouldn't it be L - 1?
        chisq = (chisq0 - (amp0 ** 2) * self._kernel_normalization) / (self._length - 2)

        return amp0, chisq

    def _get_allowed_shifts(self, allowed_shifts):
        """
        If allowed_shifts value is equal to the default UNSPECIFIED value, allowed_shifts saved in the OF object are
        returned. Otherwise, provided allowed_shifts are returned
        :param allowed_shifts: tuple of allowed shift range, None, or UNSPECIFIED. If None, any shift is allowed.
        :return: tuple of allowed shift range or None if all shifts are allowed
        """
        if allowed_shifts != UNSPECIFIED and allowed_shifts is not None and allowed_shifts[1] < allowed_shifts[0]:
            ValueError(f'Invalid allowed_shift_range: {allowed_shifts}')
        return allowed_shifts if allowed_shifts != UNSPECIFIED else self._allowed_shifts

    def fit(self, traces, allowed_shifts=UNSPECIFIED):
        """
        Performs Optimum Filter fit to the traces with the template rolls within the specified range.
        :param traces: array of traces to fit. If OF has N templates and/or N PSDs, traces can be:
            1-dimensional: one trace, fitted with N different PSDs and/or templates. Outputs have shapes (N,).
            2-dimensional: must have shape (N, L), where L is the trace length. i-th trace is fitted with i-th PSD
                and/or template. The outputs have shapes (N,)
            3-dimensional: must have shape (M, N, L), where M is arbitrary. The outputs have shapes (M, N).
        :param allowed_shifts: tuple of allowed shift range, or None. If None, any shift is allowed. If this argument is
        not provided, allowed_shifts set on the object initialization or with set_allowed_shifts method are used.
        :return: tuple of fit results: (amplitudes, time_shifts, reduced_chi_squared_values). The time shifts are in
        samples.
        """
        self._check_traces_shape(traces)
        allowed_shifts = self._get_allowed_shifts(allowed_shifts)

        # if the traces dimensionality is more than 2, while the number of PSDs and templates is 1, we flatten the
        # input array of traces. We will "unflatten" the results right before returning them
        shape = traces.shape
        if len(shape) > 2 and len(self._kernel_fft.shape) == 1:
            traces = traces.reshape(-1, shape[-1])  # flatten the traces except the last dimension

        trace_fft = scipy.fft.fft(traces, axis=-1) / self._sampling_frequency
        trace_filtered = self._kernel_fft * trace_fft / self._kernel_normalization[..., np.newaxis]
        trace_filtered_td = np.real(
            scipy.fft.ifft(trace_filtered, axis=-1)
        ) * self._sampling_frequency
        # one part of the chi2
        chi0 = np.real(
            np.einsum('...j, ...j -> ...',
                      trace_fft.conjugate() / self._noise_psd_unfolded,
                      trace_fft) * self._sampling_frequency / self._length
        )

        chit_withdelay = (trace_filtered_td ** 2) * self._kernel_normalization[..., np.newaxis]
        chi = chi0[..., np.newaxis] - chit_withdelay

        if allowed_shifts is not None:
            if allowed_shifts[0] <= 0 and allowed_shifts[1] >= -1:
                chi[..., allowed_shifts[1] + 1: allowed_shifts[0] + self._length] = np.inf
            elif allowed_shifts[0] > 0 and allowed_shifts[1] > 0:
                chi[..., :allowed_shifts[0]] = np.inf
                chi[..., allowed_shifts[1] + 1:] = np.inf
            elif allowed_shifts[0] < 0 and allowed_shifts[1] < -1:
                chi[..., :allowed_shifts[0]] = np.inf
                chi[..., allowed_shifts[1] + 1:] = np.inf
            else:
                raise ValueError(f'Invalid allowed_shift_range: {allowed_shifts}')

        best_ind = np.argmin(chi, axis=-1)
        n_dof = self._length - 3  # number of degrees of freedom. Is it correct? Not L - 2?

        # dealing with various kernel and input shapes
        # if one kernel
        if len(trace_filtered_td.shape) == 1:
            amp = trace_filtered_td[best_ind]
            chisq = chi[best_ind] / n_dof
            t0 = best_ind if best_ind < self._length // 2 else best_ind - self._length
        else:
            # many kernels, 3-dimensional traces
            if len(trace_filtered_td.shape) == 3:
                i, j = np.indices(best_ind.shape)
                amp = trace_filtered_td[i, j, best_ind]
                chisq = chi[i, j, best_ind] / n_dof
            # many kernels, one or many traces
            else:
                i = np.arange(trace_filtered_td.shape[0])
                amp = trace_filtered_td[i, best_ind]
                chisq = chi[i, best_ind] / n_dof
            t0 = best_ind
            # if shifts are larger than L/2, make them negative by subtracting L from them.
            # They are not shifts, they are actually template rolls.
            if np.any(best_ind >= self._length // 2):
                t0[best_ind >= self._length // 2] = best_ind[best_ind >= self._length // 2] - self._length

        # if the input was flattened, "unflatten" the results
        if len(shape) > 2 and len(self._kernel_fft.shape) == 1:
            amp = amp.reshape(shape[:-1])
            chisq = chisq.reshape(shape[:-1])
            t0 = t0.reshape(shape[:-1])

        return amp, t0, chisq


@export
class TwoTemplatesOptimumFilter:
    """
    Class to perform two-template Optimum Filter fits, that is two different templates with a varying shift between them
    are fitted simultaneously to the trace. Inspired by QETpy
    If noise_psd and templates are 1-dimensional, traces can be any-dimensional.
    If noise_psd or templates are 2-dimensional (i.e., N PSDs or N(*2) templates, or both N PSDs and N(*2) templates are
    provided), traces can be:
        1-dimensional: one trace, fitted with N different PSDs and/or templates. Outputs have shapes (N,)
        2-dimensional: must have shape (N, L), where L is the trace length. i-th trace is fitted with i-th PSD and/or
            template. The outputs have shapes (N,)
        3-dimensional: must have shape (M, N, L), where M is arbitrary. The outputs have shapes (M, N).
    """
    def __init__(self, template1, template2, noise_psd, sampling_frequency, allowed_shifts=None, allowed_deltas=None):
        """
       Makes an OF object to perform two-template Optimum Filter fits, that is two different templates with a varying
       time shift between them are fitted simultaneously to the traces. Inspired by QETpy.
       If noise_psd and templates are 1-dimensional, traces can be any-dimensional.
       If noise_psd or templates is 2-dimensional (i.e., N PSDs or N(*2) templates, or both N PSDs and N(*2) templates
       are provided), traces can be:
           1-dimensional: one trace, fitted with N different PSDs and/or templates. Outputs have shapes (N,)
           2-dimensional: must have shape (N, L), where L is the trace length. i-th trace is fitted with i-th PSD and/or
               template. The outputs have shapes (N,)
           3-dimensional: must have shape (M, N, L), where M is arbitrary. The outputs have shapes (M, N).
       CAUTION: specify some restrictions on the allowed template shifts and/or allowed differences of the shifts.
       Without restrictions, the fits are very slow.

       :param template1: 1- or 2-dimensional array with either one template or N templates of length L. First signal
       template
       :param template2: 1- or 2-dimensional array with either one template or N templates of length L. Second signal
       template
       :param noise_psd: 1- or 2-dimensional arrays of folded noise PSDs. If both template and noise_psd are
       2-dimensional, their first dimension must match.
       :param sampling_frequency: sampling frequency in Hz
       :param allowed_shifts: a list of tuples of allowed range of templates' rolls in time samples. If None, any rolls
       are allowed. If only one tuple is specified, it is applied to both templates.
       :param allowed_deltas: a tuple representing the range of allowed differences between the rolls of the two
       templates.
       """
        self._template1 = template1
        self._template2 = template2
        self._noise_psd = noise_psd
        self._allowed_shifts = allowed_shifts
        self._allowed_deltas = allowed_deltas
        self._check_shapes()
        self._sampling_frequency = sampling_frequency
        self._update_state()

    def set_templates(self, template1, template2):
        """
        Update the fit template
        :param template1: 1- or 2-dimensional array with either one template or N templates of length L for the first
        signal
        :param template2: 1- or 2-dimensional array with either one template or N templates of length L for the second
        signal
        """
        self._template1 = template1
        self._template2 = template2
        self._check_shapes()
        self._update_state()

    def set_noise_psd(self, noise_psd):
        """
        Update the noise PSD
        :param noise_psd: 1- or 2-dimensional arrays of folded noise PSDs. If both template and noise_psd are
        2-dimensional, their first dimension must match.
        """
        self._noise_psd = noise_psd
        self._check_shapes()
        self._update_state()

    def set_allowed_shifts(self, allowed_shifts, allowed_deltas):
        """
        Updated allowed range of template rolls and allowed difference between the template rolls
        :param allowed_shifts: a list of tuples of allowed range of templates' rolls in time samples. If None, any rolls
        are allowed. If only one tuple is specified, it is applied to both templates.
        :param allowed_deltas: a tuple representing the range of allowed differences between the rolls of the two
        templates.
        """
        self._allowed_shifts = allowed_shifts
        self._allowed_deltas = allowed_deltas
        self._check_shapes()

    def _update_state(self):
        """
        A function that recalculates the state of the OF object (i.e. the filter kernel and some other things)
        """
        self._length = self._template1.shape[-1]
        self._noise_psd_unfolded = hx.unfold_psd(self._noise_psd)

        if np.any(np.sum(self._noise_psd_unfolded[..., 1:], axis=-1) == 0):
            warn('A zero PSD was passed to the OF. Disabling noise deweighting.')
            if len(self._noise_psd_unfolded.shape) > 1:
                self._noise_psd_unfolded[np.sum(self._noise_psd_unfolded[..., 1:], axis=-1) == 0] = (
                    np.ones(self._noise_psd_unfolded.shape[-1]))
            else:
                self._noise_psd_unfolded = np.ones(self._noise_psd_unfolded.shape[-1])
        elif np.any(self._noise_psd_unfolded[..., 1:] == 0):
            warn('One of the PSD values in the OF is 0. It will cause a division by 0 error')

        self._noise_psd_unfolded[..., 0] = np.inf

        self._template1_fft = scipy.fft.fft(self._template1) / self._sampling_frequency
        self._template2_fft = scipy.fft.fft(self._template2) / self._sampling_frequency

        self._kernel1_fft = self._template1_fft.conjugate() / self._noise_psd_unfolded
        self._kernel2_fft = self._template2_fft.conjugate() / self._noise_psd_unfolded

        self._kernel1_normalization = np.real(
            np.einsum('...j, ...j -> ...', self._kernel1_fft,
                      self._template1_fft)) * self._sampling_frequency / self._length
        self._kernel2_normalization = np.real(
            np.einsum('...j, ...j -> ...', self._kernel2_fft,
                      self._template2_fft)) * self._sampling_frequency / self._length

        self._p = np.zeros(self._kernel1_fft.shape + (2, 2))

        # einsum returns a writeable view, so we can assign values to it
        # setting 1 to the diagonal elements over the last two axes
        np.einsum('...ii -> ...i', self._p)[:] = 1

        self._p[..., 0, 1] = self._p[..., 1, 0] = np.real(
            scipy.fft.ifft(self._template2_fft * self._kernel1_fft) * self._sampling_frequency
        )
        self._p[..., 0, 0] = self._kernel1_normalization[..., np.newaxis]
        self._p[..., 1, 1] = self._kernel2_normalization[..., np.newaxis]

        self._p_inv = np.linalg.pinv(self._p)

        self._resolution1 = 1 / np.sqrt(self._kernel1_normalization)
        self._resolution2 = 1 / np.sqrt(self._kernel2_normalization)

    def _check_traces_shape(self, traces):
        """
        Checks whether the traces shape correspond to the OF templates and PSDs
        :param traces: array of traces to fit
        """
        if traces.shape[-1] != self._template1.shape[-1]:
            raise ValueError(
                f'In the OF, traces length {(traces.shape[-1])} must be equal to the template length '
                f'({self._template1.shape[-1]}).')

        if (len(self._kernel1_fft.shape) > 1 and len(traces.shape) > 1
                and self._kernel1_fft.shape[0] != traces.shape[-2]):
            raise ValueError(
                f'In OF with more than one PSD or template, the -2 axis of traces should be equal to the number of '
                f'PSDs or templates. Number of PSDs is {self._noise_psd_unfolded.shape[0]}. '
                f'Number of templates is {self._template1.shape[0]}. The provided traces shape is {traces.shape}')

    def _check_shapes(self):
        """
        Checks whether the shapes of OF templates and PSDs match
        """
        if self._template1.shape != self._template2.shape:
            raise NotImplementedError(
                f'Both templates must have the same dimensions. Your template shapes are {self._template1.shape} and '
                f'{self._template2.shape}.')
        if len(self._noise_psd.shape) > 2:
            raise NotImplementedError(
                f'PSDs with more than 2 dimensions are not supported. Your PSD shape is {self._noise_psd.shape}.')
        if len(self._template1.shape) > 2 or len(self._template2.shape) > 2:
            raise NotImplementedError(
                f'Templates with more than 2 dimensions are not supported. Your template shapes are '
                f'{self._template1.shape} and {self._template2.shape}.')
        if len(self._noise_psd.shape) > 1 and len(self._template1.shape) > 1 and self._noise_psd.shape[0] != \
                self._template1.shape[0]:
            raise NotImplementedError(
                f'If templates and PSDs are 2-dimensinonal, their lengths along the 0-axis must match. You provided '
                f'{len(self._template1)} templates and {len(self._noise_psd)} PSDs.')
        if self._allowed_shifts is not None and not hasattr(self._allowed_shifts[0], '__iter__'):
            self._allowed_shifts = [self._allowed_shifts, self._allowed_shifts]
        if self._allowed_shifts is not None and (
                self._allowed_shifts[0][1] < self._allowed_shifts[0][0] or self._allowed_shifts[1][1] <
                self._allowed_shifts[1][0]):
            raise ValueError(f'Invalid allowed_shifts: {self._allowed_shifts}.')

    @property
    def resolution1(self):
        """
        Expected energy resolution of the first signal, based on the template1 and the noise PSD. Has the length of N,
        if either N templates1 or N PSDs are provided
        """
        return self._resolution1

    @property
    def resolution2(self):
        """
        Expected energy resolution of the second signal, based on the template2 and the noise PSD. Has the length of N,
        if either N templates2 or N PSDs are provided
        """
        return self._resolution2

    def _get_allowed_shifts_and_deltas(self, allowed_shifts, allowed_deltas):
        """
        If allowed_shifts and/or allowed_deltas are equal to the default UNSPECIFIED value, allowed_shifts and/or
        allowed_deltas saved in the OF object are returned. Otherwise, provided allowed_shifts and allowed_deltas are
        returned
        :param allowed_shifts: list of tuples of allowed shift ranges, None, or UNSPECIFIED. If None, any shift is
        allowed.
        :param allowed_deltas: tuple of allowed differences between the shifts of the two templates
        :return: tuple of (allowed_shifts, allowed_deltas)
        """
        if allowed_shifts != UNSPECIFIED and allowed_shifts is not None and not hasattr(allowed_shifts[0], '__iter__'):
            allowed_shifts = [allowed_shifts, allowed_shifts]
        if allowed_shifts != UNSPECIFIED and allowed_shifts is not None and (
                allowed_shifts[0][1] < allowed_shifts[0][0] or allowed_shifts[1][1] < allowed_shifts[1][0]):
            raise ValueError(f'Invalid allowed_shifts: {allowed_shifts}')

        if allowed_shifts == UNSPECIFIED:
            allowed_shifts = self._allowed_shifts
        if allowed_deltas == UNSPECIFIED:
            allowed_deltas = self._allowed_deltas

        return allowed_shifts, allowed_deltas

    def fit(self, traces, allowed_shifts=UNSPECIFIED, allowed_deltas=UNSPECIFIED):
        """
        Fit the traces with two-template Optimum Filter. If specified or if set by the init or setter function,
        allowed_shifts and allowed_deltas restrict how much the templates can roll on their own and relative to one
        another.
        :param traces: array of traces to fit. If OF has N(*2) templates and/or N PSDs, traces can be:
            1-dimensional: one trace, fitted with N different PSDs and/or templates. Outputs have shapes (N,).
            2-dimensional: must have shape (N, L), where L is the trace length. i-th trace is fitted with i-th PSD
                and/or templates. The outputs have shapes (N,)
            3-dimensional: must have shape (M, N, L), where M is arbitrary. The outputs have shapes (M, N).

        :param allowed_shifts: tuple of allowed shift range, or None. If None, any shift is allowed. If this argument is
        not provided, allowed_shifts set on the object initialization or with set_allowed_shifts method are used.
        :return: tuple of fit results: (amplitudes, time_shifts, reduced_chi_squared_values). The time shifts are in
        samples.

        :param allowed_shifts: a list of tuples of allowed range of templates' rolls in time samples. If None, any rolls
        are allowed. If only one tuple is specified, it is applied to both templates. If this argument is not provided,
        allowed_shifts set on the object initialization or with set_allowed_shifts method are used.
        :param allowed_deltas: a tuple representing the range of allowed relative rolls of the two templates. If this
        argument is not provided, allowed_deltas set on the object initialization or with set_allowed_shifts method are
        used.
        """
        self._check_traces_shape(traces)
        allowed_shifts, allowed_deltas = self._get_allowed_shifts_and_deltas(allowed_shifts, allowed_deltas)

        # if the traces dimensionality is more than 2, while the number of PSDs and templates is 1, we flatten the
        # input array of traces. We will "unflatten" the results right before returning them
        shape = traces.shape
        if len(shape) > 2 and len(self._kernel1_fft.shape) == 1:
            traces = traces.reshape(-1, shape[-1])  # flatten the traces except the last dimension

        trace_fft = scipy.fft.fft(traces, axis=-1) / self._sampling_frequency  # V
        # Q-vectors are time-domain traces filtered by the two filter kernels
        q1 = np.real(scipy.fft.ifft(trace_fft * self._kernel1_fft, axis=-1)) * self._sampling_frequency
        q2 = np.real(scipy.fft.ifft(trace_fft * self._kernel2_fft, axis=-1)) * self._sampling_frequency

        # one part of the chi2
        chi0 = np.real(
            np.einsum('...j, ...j -> ...', trace_fft.conjugate() / self._noise_psd_unfolded,
                      trace_fft) * self._sampling_frequency / self._length
        )

        n_deltas = self._length if allowed_deltas is None else allowed_deltas[1] - allowed_deltas[0] + 1
        # number of allowed shifts of the second template
        n_t2s = self._length if allowed_shifts is None else allowed_shifts[1][1] - allowed_shifts[1][0] + 1
        # if number of allowed deltas is smaller, it is a stricter constrain. We will use it instead of the second
        # template shifts
        use_deltas = n_deltas < n_t2s
        if use_deltas:
            n_t2s = n_deltas

        # Constructing flattened arrays of all the combinations of allowed first and seconds

        # all allowed rolls of the first template, repeated n times, where n is the number of allowed shifts of the
        # second template.
        t1s = np.tile(np.arange(-self._length // 2, self._length // 2), n_t2s) if allowed_shifts is None else \
            np.tile(np.arange(allowed_shifts[0][0], allowed_shifts[0][1] + 1), n_t2s)

        # number of unique shifts of the first template
        n_t1s = len(t1s) // n_t2s
        # all allowed rolls of the second template, taking into account restrictions on the relative template rolls
        # (i.e. deltas)
        if use_deltas:
            t2s = t1s + np.repeat(np.arange(allowed_deltas[0], allowed_deltas[1] + 1), n_t1s)
            t2s[t2s < -self._length] = t2s[t2s < -self._length] + self._length
            t2s[t2s >= self._length] = t2s[t2s >= self._length] - self._length
        else:
            t2s = np.repeat(np.arange(-self._length // 2, self._length // 2), n_t1s) if allowed_shifts is None else \
                np.repeat(np.arange(allowed_shifts[1][0], allowed_shifts[1][1] + 1), n_t1s)

        # if restriction on the relative rolls wasn't applied yet, applying it now
        if allowed_deltas is not None and not use_deltas:
            mask = (t2s - t1s >= allowed_deltas[0]) & (t2s - t1s <= allowed_deltas[1])
            t1s = t1s[mask]
            t2s = t2s[mask]

        p_inv = self._p_inv[..., t1s - t2s, :, :]

        a1s = p_inv[..., 0, 0] * q1[..., t1s] + p_inv[..., 0, 1] * q2[..., t2s]
        a2s = p_inv[..., 1, 0] * q1[..., t1s] + p_inv[..., 1, 1] * q2[..., t2s]

        chi2s = chi0[..., np.newaxis] - q1[..., t1s].conjugate() * a1s - q2[..., t2s].conjugate() * a2s

        best_ind = np.argmin(chi2s, axis=-1)
        n_dof = self._length - 5  # number of degrees of freedom. Should it be L - 4? I don't know...

        if len(q1.shape) == 1:
            return a1s[best_ind], a2s[best_ind], t1s[best_ind], t2s[best_ind], chi2s[best_ind] / n_dof
        else:
            if len(a1s.shape) == 3:
                i, j = np.indices(best_ind.shape)
                results = (a1s[i, j, best_ind], a2s[i, j, best_ind], t1s[best_ind], t2s[best_ind], chi2s[i, j, best_ind]
                           / n_dof)
            else:
                i = np.arange(a1s.shape[0])
                results = (a1s[i, best_ind], a2s[i, best_ind], t1s[best_ind], t2s[best_ind], chi2s[i, best_ind] / n_dof)

            if len(shape) > 2 and len(self._kernel1_fft.shape) == 1:
                results = tuple([r.reshape(shape[:-1]) for r in results])
            return results


# This is how I was testing all the vectorization features. Ideally, it should be converted into a unit-test

# f, psd = get_pink_psd(trace_length, sampling_dt, 30)
# template = get_analytical_template()
# trace = 1000*template + generate_noise(1, psd, sampling_frequency)[0]
#
# print('one trace, one PSD, one template')
# of = OptimumFilter(template, psd, sampling_frequency)
# print(of.fit_with_no_shift(trace))
# print(of.fit(trace))
# print(of.resolution)
#
# traces = np.tile(trace, 2).reshape(2, -1)
# traces[1] -= 500*template
#
# print()
# print('N traces, one PSD, one template')
# print(of.fit_with_no_shift(traces))
# print(of.fit(traces))
# print(of.resolution)
#
# traces_nm = np.tile(trace, 6).reshape(3, 2, -1)
# traces_nm[:, 1] -= 500*template
# traces_nm[0, :] -= 100*template
#
# print()
# print('MxN traces, one PSD, one template')
# print(of.fit_with_no_shift(traces_nm))
# print(of.fit(traces_nm))
# print(of.resolution)
#
# psds = np.tile(psd, 2).reshape(2, -1)
# psds[1] *= 0.5
# of = OptimumFilter(template, psds, sampling_frequency)
#
# print()
# print('one trace, N PSD, one template')
# print(of.fit_with_no_shift(trace))
# print(of.fit(trace))
# print(of.resolution)
#
# print()
# print('N traces, N PSDs, one template')
# print(of.fit_with_no_shift(traces))
# print(of.fit(traces))
# print(of.resolution)
#
# print()
# print('MxN traces, N PSD, one template')
# print(of.fit_with_no_shift(traces_nm))
# print(of.fit(traces_nm))
# print(of.resolution)
#
#
#
# f, psd = get_pink_psd(trace_length, sampling_dt, 30)
# template = get_analytical_template()
# trace = 1000*template + generate_noise(1, psd, sampling_frequency)[0]
#
# templates = np.tile(template, 2).reshape(2, -1)
# templates[1] = templates[1]/10
#
# print('one trace, one PSD, N templates')
# of = OptimumFilter(templates, psd, sampling_frequency)
# print(of.fit_with_no_shift(trace))
# print(of.fit(trace))
# print(of.resolution)
#
# traces = np.tile(trace, 2).reshape(2, -1)
# traces[1] -= 500*template
#
# print()
# print('N traces, one PSD, N templates')
# print(of.fit_with_no_shift(traces))
# print(of.fit(traces))
# print(of.resolution)
#
# traces_nm = np.tile(trace, 6).reshape(3, 2, -1)
# traces_nm[:, 1] -= 500*template
# traces_nm[0, :] -= 100*template
#
# print()
# print('MxN traces, one PSD, N templates')
# print(of.fit_with_no_shift(traces_nm))
# print(of.fit(traces_nm))
# print(of.resolution)
#
# psds = np.tile(psd, 2).reshape(2, -1)
# psds[1] *= 0.5
# of = OptimumFilter(templates, psds, sampling_frequency)
#
# print()
# print('one trace, N PSD, N templates')
# print(of.fit_with_no_shift(trace))
# print(of.fit(trace))
# print(of.resolution)
#
# print()
# print('N traces, N PSDs, N templates')
# print(of.fit_with_no_shift(traces))
# print(of.fit(traces))
# print(of.resolution)
#
# print()
# print('MxN traces, N PSD, N templates')
# print(of.fit_with_no_shift(traces_nm))
# print(of.fit(traces_nm))
# print(of.resolution)
#
#
#
#
# f, psd = get_pink_psd(trace_length, sampling_dt, 30)
# template1 = get_analytical_template()
# template2 = get_analytical_template(rise_time=200000, fall_time=8000000, prepulse_length=2148)
#
# trace = generate_noise(1, psd, sampling_frequency)[0] + 1000*np.roll(template1, -10) + 500*np.roll(template2, 3)
#
# print('one trace, one PSD, one template')
# of = TwoTemplatesOptimumFilter(template1, template2, psd, sampling_frequency, allowed_shifts=[(-10, 5), (-10, 5)], allowed_deltas=(-15,15))
# print(of.fit(trace))
# print(of.resolution1)
# print(of.resolution2)
#
# traces = np.tile(trace, 2).reshape(2, -1)
# traces[1] -= 200*np.roll(template1, -10) + 200*np.roll(template2, 3)
#
# print()
# print('N traces, one PSD, one template')
# print(of.fit(traces))
# print(of.resolution1)
# print(of.resolution2)
#
# traces_nm = np.tile(trace, 6).reshape(3, 2, -1)
# traces_nm[:, 1] -= 200*np.roll(template1, -10) + 200*np.roll(template2, 3)
# traces_nm[0, :] -= 100*np.roll(template1, -10) + 100*np.roll(template2, 3)
#
# print()
# print('MxN traces, one PSD, one template')
# print(of.fit(traces_nm))
# print(of.resolution1)
# print(of.resolution2)
#
# psds = np.tile(psd, 2).reshape(2, -1)
# psds[1] *= 0.5
# of = TwoTemplatesOptimumFilter(template1, template2, psds, sampling_frequency, allowed_shifts=[(-10, 5), (-10, 5)], allowed_deltas=(-15,15))
#
# print()
# print('one trace, N PSD, one template')
# print(of.fit(trace))
# print(of.resolution1)
# print(of.resolution2)
#
# print()
# print('N traces, N PSDs, one template')
# print(of.fit(traces))
# print(of.resolution1)
# print(of.resolution2)
#
# print()
# print('MxN traces, N PSD, one template')
# print(of.fit(traces_nm))
# print(of.resolution1)
# print(of.resolution2)
#
#
#
# f, psd = get_pink_psd(trace_length, sampling_dt, 30)
# template1 = get_analytical_template()
# template2 = get_analytical_template(rise_time=200000, fall_time=8000000, prepulse_length=2148)
#
# trace = generate_noise(1, psd, sampling_frequency)[0] + 1000*np.roll(template1, -10) + 500*np.roll(template2, 3)
#
# templates1 = np.tile(template1, 2).reshape(2, -1)
# templates1[1] = templates1[1]/10
# templates2 = np.tile(template2, 2).reshape(2, -1)
# templates2[1] = templates2[1]/10
#
# print('one trace, one PSD, N templates')
# of = TwoTemplatesOptimumFilter(templates1, templates2, psd, sampling_frequency, allowed_shifts=[(-10, 5), (-10, 5)], allowed_deltas=(-15,15))
# print(of.fit(trace))
# print(of.resolution1)
# print(of.resolution2)
#
# traces = np.tile(trace, 2).reshape(2, -1)
# traces[1] -= 200*np.roll(template1, -10) + 200*np.roll(template2, 3)
#
# print()
# print('N traces, one PSD, N templates')
# print(of.fit(traces))
# print(of.resolution1)
# print(of.resolution2)
#
# traces_nm = np.tile(trace, 6).reshape(3, 2, -1)
# traces_nm[:, 1] -= 200*np.roll(template1, -10) + 200*np.roll(template2, 3)
# traces_nm[0, :] -= 100*np.roll(template1, -10) + 100*np.roll(template2, 3)
#
# print()
# print('MxN traces, one PSD, N templates')
# print(of.fit(traces_nm))
# print(of.resolution1)
# print(of.resolution2)
#
# psds = np.tile(psd, 2).reshape(2, -1)
# psds[1] *= 0.5
# of = TwoTemplatesOptimumFilter(templates1, templates2, psds, sampling_frequency, allowed_shifts=[(-10, 5), (-10, 5)], allowed_deltas=(-15,15))
#
# print()
# print('one trace, N PSD, N templates')
# print(of.fit(trace))
# print(of.resolution1)
# print(of.resolution2)
#
# print()
# print('N traces, N PSDs, N templates')
# print(of.fit(traces))
# print(of.resolution1)
# print(of.resolution2)
#
# print()
# print('MxN traces, N PSD, N templates')
# print(of.fit(traces_nm))
# print(of.resolution1)
# print(of.resolution2)
