import numpy as np
import strax
import scipy

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
def psd_to_trace_length(psd_length, trace_length_is_even=True):
    """
    Returns the trace length corresponding to the provided length of the folded PSD
    :param psd_length: length of the folded PSD array
    :param trace_length_is_even: whether the trace length is even.
    :return: trace length corresponding to the PSD
    """
    return (psd_length-1)*2 if trace_length_is_even else (psd_length-1)*2 + 1


@export
def trace_to_psd_length(trace_length):
    """
    Returns the length of the folded PSD for the specified trace length
    :param trace_length: trace length in time samples
    :return: lenght of folded PSD
    """
    return trace_length//2 + 1


@export
def unfold_psd(psd, trace_length_is_even=True):
    """
    Unfolds folded PSD. (PSDs for real-valued traces are symmetrical, therefore they can be folded in half.)
    :param psd: folded PSD. Can be multidimensional
    :param trace_length_is_even: whether the trace length is even. This defines whether folded PSD has an unpaired
    frequency component corresponding to the maximal frequency
    :return: unfodled PSD
    """
    if trace_length_is_even:
        return np.concatenate((psd[..., 0:1],
                               psd[..., 1:-1] / 2,
                               psd[..., -1:],
                               psd[..., -2:0:-1] / 2), axis=-1)
    else:
        return np.concatenate((psd[..., 0],
                               psd[..., 1:] / 2,
                               psd[..., -1:0:-1] / 2), axis=-1)


@export
def calculate_csd(traces_a, traces_b, sampling_frequency=1.0):
    """
    Calculates the real part of the Cross-Spectral Densities of two sets of traces.
    Adapted from https://github.com/spice-herald/QETpy/blob/master/qetpy/core/_noise.py

    :param traces_a: first set of traces. n-dimensional: (n1, n2, ..., nm, n_traces_per_CSD, trace_length)
    :param traces_b: second set of traces. n-dimensional: (n1, n2, ..., nm, n_traces_per_CSD, trace_length)
    :param sampling_frequency: sampling frequency in Hz
    :return: (f, csd) - frequency components and folded real parts of the CSDs of shape (n1, ..., nm, trace_length//2+1)
    """

    norm = sampling_frequency * traces_a.shape[-1]
    traces_a_f = scipy.fft.rfft(traces_a)
    traces_b_f = scipy.fft.rfft(traces_b)
    ab_star = np.real(traces_a_f * traces_b_f.conjugate())  # Re(V_a V_b*)

    if len(ab_star.shape) == 1:
        csd = ab_star / norm
    else:
        csd = np.mean(ab_star, axis=0) / norm

    # multiply the necessary frequencies by two (zeroth frequency should be the same, as
    # should the last frequency when x.shape[-1] is odd)
    csd[..., 1:traces_a.shape[-1] // 2 + 1 - (traces_a.shape[-1] + 1) % 2] *= 2.0
    f = scipy.fft.rfftfreq(traces_a.shape[-1], d=1.0 / sampling_frequency)

    return f, csd


@export
def calculate_psd(traces, sampling_frequency=1.0):
    """
    Calculates the Power Spectral Densities of traces.

    :param traces: array of traces. n-dimensional: (n1, n2, ..., nm, n_traces_per_PSD, trace_length)
    :param sampling_frequency: sampling frequency in Hz
    :return: (f, psd) - frequency components and folded PSDs of shape (n1, ..., nm, trace_length//2+1)
    """
    return calculate_csd(traces, traces, sampling_frequency)


@export
def get_csd_index(n_channels, channel_indices):
    """
    Takes a pair of channel indices and returns a CSD index corresponding to the CSD of this pair.
    For a set of channels, the number of all the possible CSDs is not n_channels^2, but n_channels*(n_channels - 1)/2.
    Therefore, to save memory, it is convinient to save CSDs in a 2d-array (n_csds, csd_length) rather than in a
    3d-array (n_channels, n_channels, csd_length).

    :param n_channels: number of channels in the channel map
    :param channel_indices: a 2d array of pairs of channel indices. Or a 1d-array with one pair.
    :return: an array of CSD indices or one CSD index if channel_indices is 1d
    """
    sorted_indices = np.sort(channel_indices, axis=-1)
    one_input = (len(sorted_indices.shape) == 1)

    if one_input:
        sorted_indices = np.array([sorted_indices, ])

    i = sorted_indices[:, 0]
    j = sorted_indices[:, 1]

    # Check if indices are within bounds and valid for the upper triangular matrix excluding the diagonal
    if np.any(i < 0) or np.any(j >= n_channels) or np.any(i == j):
        raise ValueError("Invalid channel indices. Indices cannot be equal or outside of range [0, n_channels)")

    # Calculate the index in the flattened upper triangular matrix excluding the diagonal
    csd_indices = (i * n_channels - (i * (i + 1)) // 2) + (j - i - 1)
    return csd_indices[0] if one_input else csd_indices


@export
def get_all_growing_pairs(x):
    """
    Returns all the possible (A, B) pairs of elements such that A < B
    :param x: list of elements
    :return: array of all the possible growing pairs
    """
    pairs = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
    return pairs[pairs[:, 0] < pairs[:, 1]]


@export
def calculate_psd_of_sum(psds, csds, channel_mask):
    """
    Calculates a PSD of the sum of channels form individual channel PSDs and their CSDs
    :param psds: channel PSDs
    :param csds: CSDs of all the unique channel pairs sorted with by the get_csd_index function
    :param channel_mask: mask of channels to sum
    :return: PSD of the sum of channels
    """
    if channel_mask.sum() == 0:
        raise ValueError('At least one channel should be True in channel_mask in calculate_psd_of_sum()')

    channel_indices = np.where(channel_mask)[0]
    scd_indices = get_csd_index(len(channel_mask), get_all_growing_pairs(channel_indices))
    return np.sum(psds[channel_indices], axis=0) + 2 * np.sum(csds[scd_indices], axis=0)
