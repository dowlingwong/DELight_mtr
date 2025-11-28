"""
Energy sweep runner with dual OptimumFilters:
- First 19 channels use submerged-channel template
- Last 37 channels use vacuum-channel template

Hard-coded paths and parameters:
    in_pattern   = /ceph/dwong/trigger_samples/v2_300k/traces_energy_{energy}.zst
    sub_template = /home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/sub_ch_template.npy
    vac_template = /home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy
    noise_psd    = /home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy
    fs           = 3906250
    repeats      = 100
    n_traces     = 56
    samples      = 300000
    energies     = 5, 10, 15, 20, 25, 30
    out_dir      = ./max_amp
"""

from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import zstandard as zstd

from wk27.OptimumFilter import OptimumFilter

# -------------------
# Worker process globals
# -------------------
_WORKER_OF_SUB = None
_WORKER_OF_VAC = None
_REANCHOR_EVERY = None


def _worker_init(sub_template: np.ndarray, vac_template: np.ndarray,
                 noise_psd: np.ndarray, fs: float, reanchor_every: int | None):
    """Initializer runs once per worker. Builds two OptimumFilters."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    global _WORKER_OF_SUB, _WORKER_OF_VAC, _REANCHOR_EVERY
    _WORKER_OF_VAC = OptimumFilter(vac_template, noise_psd, fs)
    _WORKER_OF_SUB = OptimumFilter(sub_template, noise_psd, fs)
    sum_psd = noise_psd*37
    _sum_OF_VAC = OptimumFilter(vac_template, sum_psd, fs)
    _REANCHOR_EVERY = reanchor_every


def _max_amp_for_trace(args) -> float:
    """Pick correct OptimumFilter depending on channel index."""
    trace_1d, ch = args
    x = np.ascontiguousarray(trace_1d, dtype=np.float64)
    if ch < 19:  # first 19 channels: submerged OF
        amps, _ = _WORKER_OF_SUB.sliding_fit(x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none")
    else:        # remaining 37 channels: vacuum OF
        amps, _ = _WORKER_OF_VAC.sliding_fit(x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none")
    return float(np.max(amps))


# -------------------
# Loader for traces from Zstandard
# -------------------
def load_traces_from_zstd(input_path: Path, n_traces: int,
                          trace_shape: tuple[int, int], dtype=np.float16) -> np.ndarray:
    """
    Load stacked ndarray of shape (n_traces, *trace_shape) from a .zst file.
    """
    def unshuffle_bytes(data: bytes, dtype, shape) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        num_elements = int(np.prod(shape))
        reshaped = np.frombuffer(data, dtype=np.uint8).reshape(itemsize, num_elements).T
        unshuffled = reshaped.reshape(-1)
        return unshuffled.view(dtype).reshape(shape)

    # --- compute expected uncompressed size ---
    itemsize = np.dtype(dtype).itemsize
    trace_size_bytes = int(np.prod(trace_shape)) * itemsize
    expected_size = n_traces * trace_size_bytes

    decompressor = zstd.ZstdDecompressor()
    with open(input_path, "rb") as f:
        compressed_content = f.read()
        # Some zstd frames don't store content size -> we must cap output size
        decompressed = decompressor.decompress(compressed_content, max_output_size=expected_size)

    if len(decompressed) != expected_size:
        raise ValueError(f"Decompressed size {len(decompressed)} != expected {expected_size}")

    traces = []
    for i in range(n_traces):
        start = i * trace_size_bytes
        end = start + trace_size_bytes
        trace_bytes = decompressed[start:end]
        trace = unshuffle_bytes(trace_bytes, dtype=dtype, shape=trace_shape)
        traces.append(trace)

    return np.stack(traces).astype(np.float32)




# -------------------
# Core computation
# -------------------
def compute_max_amplitudes(traces: np.ndarray, sub_template: np.ndarray, vac_template: np.ndarray,
                           noise_psd: np.ndarray, fs: float, *, max_workers: int = 56,
                           reanchor_every: int | None = None, show_progress: bool = True) -> np.ndarray:
    """Compute max OF amplitude per trace with concurrency. Returns (R, T)."""
    R, T, L = traces.shape
    n_workers = min(int(max_workers), os.cpu_count() or 1)
    max_vals = np.empty((R, T), dtype=np.float64)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(sub_template, vac_template, noise_psd, fs, reanchor_every),
    ) as exe:
        r_iter = range(R)
        if show_progress:
            r_iter = tqdm(r_iter, desc="Repeats", unit="repeat")
        for r in r_iter:
            gen = ((traces[r, t, :], t) for t in range(T))
            mapped = exe.map(_max_amp_for_trace, gen, chunksize=1)
            if show_progress:
                mapped = tqdm(mapped, total=T, desc=f"Repeat {r+1}/{R}", unit="trace", leave=False)
            row = [val for val in mapped]
            max_vals[r, :] = np.asarray(row, dtype=np.float64)

    return max_vals


# -------------------
# Main runner
# -------------------
def main() -> int:
    NOISE_POWERS = [13.2, 17.2, 21.3, 25.2, 29.2, 33.2, 37.2, 41.2, 45.2, 49.2, 53.2]

    sub_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/sub_ch_template.npy")
    vac_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy")
    noise_psd = np.load("/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy")

    fs = 3906250
    repeats = 200
    n_traces = 56
    samples = 150000
    energies = list(range(0, 60, 2))

    # Loop over all noise power directories
    for noise_power in NOISE_POWERS:
        noise_str = f"{noise_power:.1f}"  # e.g., "13.2"

        # Pattern for input traces
        in_pattern = f"/ceph/dwong/work/threshold/noise_{noise_str}/traces_energy_{{energy}}.zst"

        # Output directory for this noise power
        out_dir = Path(f"/ceph/dwong/work/threshold/noise_{noise_str}/indiv_max_amp")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing noise power {noise_str} ===")

        for E in tqdm(energies, desc=f"Noise {noise_str} | Energies", unit="E"):
            in_path = Path(in_pattern.format(energy=E))
            out_path = out_dir / f"max_amplitudes_energy_{E}.npy"

            if not in_path.exists():
                print(f"[skip] Input file not found: {in_path}")
                continue

            traces = load_traces_from_zstd(
                in_path,
                n_traces=repeats,
                trace_shape=(n_traces, samples),   # (56, 150000)
                dtype=np.float16,
            )
            if traces.shape != (repeats, n_traces, samples):
                raise ValueError(f"Bad shape {traces.shape}")


            max_vals = compute_max_amplitudes(
                traces=traces,
                sub_template=sub_template,
                vac_template=vac_template,
                noise_psd=noise_psd,
                fs=fs,
                max_workers=60,
                reanchor_every=None,
                show_progress=True,
            )

            np.save(out_path, max_vals)
            print(f"Saved {out_path} with shape {max_vals.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
