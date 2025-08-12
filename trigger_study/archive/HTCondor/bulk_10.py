import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import multiprocessing

from trace_IO import *
from OptimumFilter import *
# ----------------------
# Constants & Parameters
# ----------------------
energy = 10
n_sets = 100
window_size = 32768
num_traces = 9
trace_length = 250_000
num_windows = trace_length - window_size + 1

trace_path = f"/ceph/dwong/trigger_samples/lts/traces_energy_{energy}.zst"
output_path = Path(f"/ceph/dwong/trigger_samples/lts/ampl_energy_{energy}.zst")

template_path = "/home/dwong/DELight_mtr/trigger_study/wk15/templates/vac_ch_template.npy"
noise_psd_path = "/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy"
fs = 3906250

# ----------------------
# Load template and PSD once
# ----------------------
template = np.load(template_path)
noise_psd = np.load(noise_psd_path)

# ----------------------
# Process function (1 repeat = 9 traces)
# ----------------------
def process_one_repeat(traces, template, noise_psd, fs):
    vac = OptimumFilter(template, noise_psd, fs)
    result = np.empty((num_traces, num_windows), dtype=np.float32)
    window = np.empty(window_size, dtype=traces.dtype)

    for ch in range(num_traces):
        trace = traces[ch]
        ampl_arr = result[ch]
        for i in range(num_windows):
            window[:] = trace[i:i+window_size]
            ampl_arr[i] = vac.fit(window)

    return result

# ----------------------
# Main
# ----------------------
def main():
    # Load traces from file
    loaded_traces = load_traces_from_zstd(trace_path, n_traces=n_sets)  # shape: (100, 9, 250000)
    print(loaded_traces.shape)
    assert loaded_traces.shape == (n_sets, num_traces, trace_length), "Trace shape mismatch"

    # Use all available CPU cores
    n_jobs = multiprocessing.cpu_count()-2

    # Parallel amplitude extraction
    all_ampls = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_one_repeat)(loaded_traces[i], template, noise_psd, fs)
        for i in tqdm(range(n_sets), desc=f"Processing Energy {energy} ({n_sets} Traces)")
    )

    # Save results
    save_ampl_to_zstd(all_ampls, output_path)

if __name__ == "__main__":
    main()
