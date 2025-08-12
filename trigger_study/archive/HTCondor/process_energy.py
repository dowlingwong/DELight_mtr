#!/usr/bin/env python3
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from trace_IO import load_traces_from_zstd, save_ampl_to_zstd
from OptimumFilter import OptimumFilter

# ----------------------
# Process function (1 repeat = 9 traces)
# ----------------------
def process_one_repeat(traces, template, noise_psd, fs, window_size):
    num_traces = traces.shape[0]
    trace_length = traces.shape[1]
    num_windows = trace_length - window_size + 1

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

def main():
    p = argparse.ArgumentParser(description="Process one energy with OptimumFilter.")
    p.add_argument("--energy", type=int, required=True, help="Energy in eV, e.g., 10, 20, ...")
    p.add_argument("--n-sets", type=int, default=100, help="Number of repeats (sets) to load/process.")
    p.add_argument("--num-traces", type=int, default=9, help="Channels per set.")
    p.add_argument("--trace-length", type=int, default=250_000, help="Samples per trace.")
    p.add_argument("--window-size", type=int, default=32768, help="Sliding window length.")
    p.add_argument("--fs", type=float, default=3_906_250, help="Sampling rate (Hz).")
    p.add_argument("--template-path", default="/home/dwong/DELight_mtr/trigger_study/wk15/templates/vac_ch_template.npy")
    p.add_argument("--noise-psd-path", default="/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy")
    p.add_argument("--trace-dir", dest="trace_dir", default="/ceph/dwong/trigger_samples/lts", help="Base dir for traces.")
    p.add_argument("--output-dir", dest="output_dir", default="/ceph/dwong/trigger_samples/lts", help="Base dir for outputs.")
    p.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 2), help="Parallel jobs.")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output zst already exists.")
    args = p.parse_args()

    # Sanity checks and logging
    assert args.window_size <= args.trace_length, "window_size > trace_length"
    print(f"Using {args.jobs} parallel jobs")

    # Basic param sanity check
    assert args.window_size <= args.trace_length, "window_size cannot be larger than trace_length"
    print(f"Using {args.jobs} parallel jobs")

    energy = args.energy
    trace_path = f"{args.trace_dir}/traces_energy_{energy}.zst"
    output_path = Path(f"{args.output_dir}/ampl_energy_{energy}.zst")

    if args.skip_existing and output_path.exists():
        print(f"[SKIP] {output_path} already exists.")
        return

    # Load template and PSD
    template = np.load(args.template_path)
    noise_psd = np.load(args.noise_psd_path)

    # Load traces
    loaded_traces = load_traces_from_zstd(trace_path, n_traces=args.n_sets)  # shape: (n_sets, 9, 250000)
    print("Loaded traces shape:", loaded_traces.shape)
    assert loaded_traces.shape == (args.n_sets, args.num_traces, args.trace_length), "Trace shape mismatch"

    # Derived
    num_windows = args.trace_length - args.window_size + 1
    print(f"Energy {energy} eV | sets={args.n_sets} | traces/set={args.num_traces} | window={args.window_size} | windows/trace={num_windows}")

    # Parallel processing
    all_ampls = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(process_one_repeat)(loaded_traces[i], template, noise_psd, args.fs, args.window_size)
        for i in tqdm(range(args.n_sets), desc=f"Processing Energy {energy} ({args.n_sets} sets)")
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ampl_to_zstd(all_ampls, output_path)
    print(f"[DONE] Saved: {output_path}")

if __name__ == "__main__":
    main()
