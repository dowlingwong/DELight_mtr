import helix as hx
from helix import units
import strax
import numpy as np
import scipy
import numba

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
def get_analytical_template(rise_time=20 * units.us, fall_time=5 * units.ms,
                            length=hx.DEFAULT_TEMPLATE_LENGTH, prepulse_length=hx.DEFAULT_PREPULSE_LENGTH,
                            sampling_dt=hx.DEFAULT_SAMPLING_DT):
    """
    Returns a template with exponential rise and fall edges
    :param rise_time: rise time in ns
    :param fall_time: fall time in ns
    :param length: trace length in samples
    :param prepulse_length: prepulse region length in samples
    :param sampling_dt: sampling time in ns
    :return: np.ndarray with template of unity amplitude
    """
    t = np.arange(2*length) * sampling_dt
    result = 1 / fall_time * np.exp(-t / fall_time) - 1 / rise_time * np.exp(-t / rise_time)
    result[result < 0] = 0
    shift = prepulse_length - np.argmax(result > 0)
    if -shift > length:
        raise NotImplementedError('Template with such a small ratio between the rise and the fall times'
                                  'cannot be produced')
    if shift > 0:
        return np.pad(result, (shift, 0))[:-length-shift] / np.max(result)
    elif shift < 0:
        return result[-shift:length-shift] / np.max(result)
    else:
        return result[:length]/np.max(result)


@export
def generate_noise(n_traces, psd, sampling_frequency=1.0):
    """
    Function to generate noise traces with random phase from a given PSD. Adapted from QETpy
    :param n_traces: int. number of traces to generate
    :param psd: ndarray. folded power spectral density used to generate the noise.
    :param sampling_frequency: float. Sample frequency in Hz.
    :returns: ndarray. An array of generated noise traces

    """
    trace_length = hx.psd_to_trace_length(len(psd))

    vals = np.random.randn(n_traces, trace_length)
    vals_fft = scipy.fft.fft(vals)
    unfolded_psd = hx.unfold_psd(psd, trace_length % 2 == 0)
    unfolded_psd[0] = 0
    noise_fft = vals_fft * np.sqrt(unfolded_psd * sampling_frequency)
    return scipy.fft.ifft(noise_fft).real


@export
@numba.njit(cache=True, nogil=True)
def threshold_trigger(record, trigger_threshold, deactivation_ratio=1, max_hit_length=None, trigger_holdoff=0):
    """
    Applies a double-threshold trigger to a trace. A trigger is issued when the trace crosses the threshold.
    Then no triggers are issued before the trace crosses teh deactivation threshold. Triggers cannot be closer one to
    another than the trigger_holdoff value

    :param record: one trace
    :param trigger_threshold: trigger threshold
    :param deactivation_ratio: ratio between deactivation and activation threshold.
    :param max_hit_length: maximum time to search for the maximum value of the trace for each trigger.
    :param trigger_holdoff: time samples after each trigger when other triggers are not allowed
    :return: (hit_max_locations, start_locations, hit_lengths, hit_amplitudes, deactivation_crossed):
        hit_max_locations: locations of maximum trace values for each trigger (here called hit)
        start_locations: location where the trace crosses the activation threshold
        hit_length: length between the activation and deactivation threshold crossings. Can't be larger than
            max_hit_length
        hit_amplitudes: maximum trace value for each trigger
        deactivation_crossed: whether the deactivation threshold was crossed. If false, either the end of the trace or
            the max_hit_length was reached before the trace crossed the deactivation threshold.
    """
    hit_starts = np.flatnonzero((record[0:-1] < trigger_threshold) & (record[1:] > trigger_threshold)) + 1

    deactivation_ratio *= trigger_threshold

    hit_max_locations = np.empty(hit_starts.shape, dtype=np.int32)
    hit_amplitudes = np.empty(hit_starts.shape, dtype=np.float32)
    hit_lengths = np.empty(hit_starts.shape, dtype=np.int64)
    start_locations = np.empty(hit_starts.shape, dtype=np.int32)
    deactivation_crossed = np.ones(hit_starts.shape, dtype=np.bool_)

    i = 0
    for hit_start in hit_starts:
        if max_hit_length is None:
            hit_length = np.argmax(record[hit_start:] <= deactivation_ratio)
            if hit_length == 0:
                hit_length = len(record) - hit_start
                deactivation_crossed[i] = False
        else:
            hit_length = np.argmax(record[hit_start:hit_start + max_hit_length] < deactivation_ratio)
            if hit_length == 0:
                hit_length = min(max_hit_length, len(record) - hit_start)
                deactivation_crossed[i] = False

        offset = np.argmax(record[hit_start:hit_start + hit_length])
        hit_max_location = hit_start + offset
        if i > 0 and hit_max_location <= hit_max_locations[i - 1] + trigger_holdoff:
            continue
        hit_max_locations[i] = hit_max_location
        hit_amplitudes[i] = record[hit_max_location]
        hit_lengths[i] = hit_length
        start_locations[i] = hit_start
        i += 1

    return hit_max_locations[:i], start_locations[:i], hit_lengths[:i], hit_amplitudes[:i], deactivation_crossed[:i]


@export
@numba.njit(cache=True, nogil=True)
def threshold_trigger_2d(records, trigger_threshold, deactivation_ratio=1, max_hit_length=None, trigger_holdoff=0):
    """
    Applies a double-threshold trigger to the traces. A trigger is issued when the trace crosses the threshold.
    Then no triggers are issued before the trace crosses teh deactivation threshold. Triggers cannot be closer one to
    another than the trigger_holdoff value

    :param records: 2d array of traces
    :param trigger_threshold: trigger threshold
    :param deactivation_ratio: ratio between deactivation and activation threshold.
    :param max_hit_length: maximum time to search for the maximum value of the trace for each trigger.
    :param trigger_holdoff: time samples after each trigger when other triggers are not allowed
    :return: list of tuples (hit_max_locations, start_locations, hit_lengths, hit_amplitudes, deactivation_crossed):
        hit_max_locations: locations of maximum trace values for each trigger (here called hit)
        start_locations: location where the trace crosses the activation threshold
        hit_length: length between the activation and deactivation threshold crossings. Can't be larger than
            max_hit_length
        hit_amplitudes: maximum trace value for each trigger
        deactivation_crossed: whether the deactivation threshold was crossed. If false, either the end of the trace or
            the max_hit_length was reached before the trace crossed the deactivation threshold.
    """
    return [threshold_trigger(r, trigger_threshold, deactivation_ratio, max_hit_length, trigger_holdoff) for r in
            records]


@export
def get_gdf_kernel(sigma, n_sigma=3):
    """
    Returns normalized Gaussian Derivative to be used as the GDF kernel.
    :param sigma: width of the Gaussian in time samples
    :param n_sigma: number of sigmas to incude in the kernel
    :return: GDF kernel as a np.array
    """
    x = np.arange(-sigma*n_sigma, sigma*n_sigma + 1)
    gaus = scipy.stats.norm(scale=sigma).pdf(x)
    kernel = np.diff(gaus)
    step_function = -np.heaviside(x[:-1], 1)
    return kernel/np.dot(kernel, step_function)
