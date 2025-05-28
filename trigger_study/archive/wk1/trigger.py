import numpy as np
from scipy import signal
import scipy.stats

def get_gdf_kernel(sigma, n_sigma=3):
    """
    Returns normalized Gaussian Derivative to be used as the GDF kernel.
    :param sigma: width of the Gaussian in time samples
    :param n_sigma: number of sigmas to include in the kernel
    :return: GDF kernel as a np.array
    """
    x = np.arange(-sigma * n_sigma, sigma * n_sigma + 1)
    gaus = scipy.stats.norm(scale=sigma).pdf(x)
    kernel = np.diff(gaus)
    step_function = -np.heaviside(x[:-1], 1)
    return kernel / np.dot(kernel, step_function)

def threshold_trigger_1d(raw_record, kernel, trigger_threshold=100, deactivation_threshold_coefficient=1, trigger_holdoff=0):
    """
    Fast threshold trigger with convolution for a single trace.
    Assumes a simple threshold crossing (with optional holdoff).
    """
    filtered = signal.convolve(raw_record, kernel, mode='valid')
    deactivate_threshold = trigger_threshold * deactivation_threshold_coefficient

    triggered = False
    hits = []
    i = 0
    while i < len(filtered):
        if not triggered and filtered[i] > trigger_threshold:
            hits.append(i)
            triggered = True
            i += trigger_holdoff  # fast skip
        elif triggered and filtered[i] < deactivate_threshold:
            triggered = False
        i += 1

    return filtered, hits

def threshold_trigger_2d(records, sigma, trigger_threshold=100, deactivation_threshold_coefficient=1, trigger_holdoff=0, n_sigma=3):
    """
    Optimized batch threshold trigger for multiple records with configurable sigma.
    Generates the GDF kernel internally using sigma.

    :param records: 2D array of traces (n_records, n_samples)
    :param sigma: Gaussian sigma used for GDF kernel
    :param trigger_threshold: Threshold for triggering
    :param deactivation_threshold_coefficient: Coefficient for deactivation threshold
    :param trigger_holdoff: Number of samples to hold off after trigger
    :param n_sigma: Number of sigmas to include in the kernel
    :return: (filtered_records, all_hits, total_hits)
    """
    kernel = get_gdf_kernel(sigma, n_sigma)
    num_records = len(records)
    filtered_records = np.empty(num_records, dtype=object)
    all_hits = []

    kernel_len = len(kernel)
    conv_len = records.shape[1] - kernel_len + 1

    for i in range(num_records):
        raw = records[i]
        filtered = signal.convolve(raw, kernel, mode='valid')
        filtered_records[i] = filtered

        deactivate_threshold = trigger_threshold * deactivation_threshold_coefficient
        hits = []
        triggered = False
        j = 0
        while j < conv_len:
            if not triggered and filtered[j] > trigger_threshold:
                hits.append(j)
                triggered = True
                j += trigger_holdoff
            elif triggered and filtered[j] < deactivate_threshold:
                triggered = False
            j += 1

        all_hits.append(hits)

    total_hits = sum(len(h) for h in all_hits)
    return filtered_records, all_hits, total_hits
