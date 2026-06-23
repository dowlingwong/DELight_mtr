#!/usr/bin/env python3
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from wk23.training_samples.trace_IO import load_traces_from_zstd, save_ampl_to_zstd
from archive.script_training_HTCondor.HTCondor.OptimumFilter import OptimumFilter

# ----------------------
# Process function (1 repeat = 9 traces)
# ----------------------
def process_one_repeat(traces, template, noise_psd, fs):
    """
    traces: array-like shape (num_traces, trace_length)
    Returns:
        amplitudes: np.ndarray shape (num_traces, num_windows)
        chisqs:     np.ndarray shape (num_traces, num_windows)
    where num_windows = trace_length - template_length + 1
    """
    num_traces = traces.shape[0]
    vac = OptimumFilter(template, noise_psd, fs)

    amps_list = []
    chisq_list = []

    for ch in range(num_traces):
        trace = traces[ch]
        # sliding_fit returns (amps, chisqs, starts) over the *entire* trace
        amps, chisqs, _ = vac.sliding_fit(trace)
        # cast to float32 to keep filesize sane (optional)
        amps_list.append(amps.astype(np.float32, copy=False))
        chisq_list.append(chisqs.astype(np.float32, copy=False))

    # All traces produce the same num_windows, so stack along axis=0
    amplitudes = np.vstack(amps_list)   # (num_traces, num_windows)
    chisqs     = np.vstack(chisq_list)  # (num_traces, num_windows)
    return amplitudes, chisqs


def main():
    p = argparse.ArgumentParser(description="Process one energy with OptimumFilter.")
    p.add_argument("--energy", type=int, required=True, help="Energy in eV, e.g., 10, 20, ...")
    p.add_argument("--n-sets", type=int, default=100, help="Number of repeats (sets) to load/process.")
    p.add_argument("--num-traces", type=int, default=9, help="Channels per set.")
    p.add_argument("--trace-length", type=int, default=250_000, help="Samples per trace.")
    p.add_argument("--window-size", type=int, default=32768, help="Template/window length (for sanity check).")
    p.add_argument("--fs", type=float, default=3_906_250, help="Sampling rate (Hz).")
    p.add_argument("--template-path", default="/home/dwong/DELight_mtr/trigger_study/wk15/templates/vac_ch_template.npy")
    p.add_argument("--noise-psd-path", default="/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy")
    p.add_argument("--trace-dir", dest="trace_dir", default="/ceph/dwong/trigger_samples/lts", help="Base dir for traces.")
    p.add_argument("--output-dir", dest="output_dir", default="/ceph/dwong/trigger_samples/lts_stft", help="Base dir for outputs.")
    p.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count()/2), help="Parallel jobs.")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output zst already exists.")
    args = p.parse_args()

    print(f"Using {args.jobs} parallel jobs")

    # Load template and PSD
    template = np.load(args.template_path)
    noise_psd = np.load(args.noise_psd_path)

    # Sanity checks
    template_len = int(template.size)
    assert args.window_size == template_len, (
        f"window_size ({args.window_size}) must equal template length ({template_len})"
    )
    assert args.window_size <= args.trace_length, "window_size cannot be larger than trace_length"

    # Paths
    energy = args.energy
    trace_path = f"{args.trace_dir}/traces_energy_{energy}.zst"
    out_dir = Path(args.output_dir)
    amp_path   = out_dir / f"ampl_energy_{energy}.zst"
    chisq_path = out_dir / f"chisq_energy_{energy}.zst"

    # Skip if both exist
    if args.skip_existing and amp_path.exists() and chisq_path.exists():
        print(f"[SKIP] {amp_path} and {chisq_path} already exist.")
        return

    # Load traces: expected shape (n_sets, num_traces, trace_length)
    loaded_traces = load_traces_from_zstd(trace_path, n_traces=args.n_sets)
    print("Loaded traces shape:", loaded_traces.shape)
    assert loaded_traces.shape == (args.n_sets, args.num_traces, args.trace_length), "Trace shape mismatch"

    # Derived (for logging)
    num_windows = args.trace_length - template_len + 1
    print(f"Energy {energy} eV | sets={args.n_sets} | traces/set={args.num_traces} "
          f"| template={template_len} | windows/trace={num_windows}")

    # Parallel processing: returns list of (amplitudes, chisqs) per set
    results = Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(process_one_repeat)(loaded_traces[i], template, noise_psd, args.fs)
        for i in tqdm(range(args.n_sets), desc=f"Processing Energy {energy} ({args.n_sets} sets)")
    )

    # Unzip list of tuples -> two lists length n_sets
    all_ampls, all_chisqs = zip(*results)  # each element is (num_traces, num_windows)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    save_ampl_to_zstd(all_ampls, amp_path)     # same saver as before; just a different filename
    save_ampl_to_zstd(all_chisqs, chisq_path)  # reuse the same saver for χ² (or use save_chisq_to_zstd if you have it)
    print(f"[DONE] Saved amplitude to: {amp_path}")
    print(f"[DONE] Saved chi-square to: {chisq_path}")


if __name__ == "__main__":
    main()
