"""
Energy sweep runner with dual OptimumFilters:
- First 19 channels use submerged-channel template
- Last 37  channels use vacuum-channel template

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

from archive.trigger_study.archive.wk27.OptimumFilter import OptimumFilter

# -------------------
# Worker process globals
# -------------------
_WORKER_OF_SUB: OptimumFilter | None = None
_WORKER_OF_VAC: OptimumFilter | None = None
_REANCHOR_EVERY: int | None = None
_FIRST_SUB_CHANNELS = 19  # first 19 channels use submerged template


def _worker_init(sub_template: np.ndarray, vac_template: np.ndarray,
                 noise_psd: np.ndarray, fs: float, reanchor_every: int | None):
    """Initializer runs once per worker. Builds two OptimumFilters."""
    # keep worker threads single-threaded to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    global _WORKER_OF_SUB, _WORKER_OF_VAC, _REANCHOR_EVERY
    _WORKER_OF_SUB = OptimumFilter(sub_template, noise_psd, fs)
    _WORKER_OF_VAC = OptimumFilter(vac_template, noise_psd, fs)
    _REANCHOR_EVERY = reanchor_every


def _max_amp_for_trace(args) -> tuple[float, int]:
    """
    Compute the sliding OF amplitudes for one (trace, channel),
    return (max_amplitude, argmax_index_within_amps).
    """
    trace_1d, ch = args
    x = np.ascontiguousarray(trace_1d, dtype=np.float64)

    # choose filter by channel index
    if ch < _FIRST_SUB_CHANNELS:
        amps, _ = _WORKER_OF_SUB.sliding_fit(
            x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none"
        )
    else:
        amps, _ = _WORKER_OF_VAC.sliding_fit(
            x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none"
        )

    # ensure numeric array
    amps = np.asarray(amps, dtype=np.float64)

    # argmax and max (index in the amps vector, not absolute sample time)
    idx = int(np.argmax(amps))
    return float(amps[idx]), idx


# -------------------
# Loader for traces from Zstandard
# -------------------
def load_traces_from_zstd(input_path: Path, n_traces: int, trace_shape: tuple[int, ...],
                          dtype=np.float16) -> np.ndarray:
    """Load stacked ndarray of shape (n_traces, *trace_shape) from a .zst file."""
    if trace_shape is None:
        raise ValueError("trace_shape must be provided.")

    def _unshuffle_bytes(data: bytes, dtype, shape):
        # inverse of bytewise shuffle (item-major to byte-major); safe fallback if used during save
        itemsize = np.dtype(dtype).itemsize
        num_elements = int(np.prod(shape))
        reshaped = np.frombuffer(data, dtype=np.uint8).reshape(itemsize, num_elements).T
        unshuffled = reshaped.reshape(-1)
        return unshuffled.view(dtype).reshape(shape)

    decompressor = zstd.ZstdDecompressor()
    with open(input_path, "rb") as f:
        decompressed = decompressor.decompress(f.read())

    trace_size_bytes = int(np.prod(trace_shape)) * np.dtype(dtype).itemsize
    expected_size = n_traces * trace_size_bytes
    if len(decompressed) != expected_size:
        raise ValueError(f"Decompressed size {len(decompressed)} != expected {expected_size}")

    traces = []
    for i in range(n_traces):
        start = i * trace_size_bytes
        end = start + trace_size_bytes
        trace_bytes = decompressed[start:end]
        trace = _unshuffle_bytes(trace_bytes, dtype=dtype, shape=trace_shape)
        traces.append(trace)

    # store in float32 to cut memory but keep precision reasonable
    return np.stack(traces).astype(np.float32)


# -------------------
# Core computation
# -------------------
def compute_max_amplitudes(
    traces: np.ndarray,
    sub_template: np.ndarray,
    vac_template: np.ndarray,
    noise_psd: np.ndarray,
    fs: float,
    *,
    max_workers: int = 56,
    reanchor_every: int | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute max OF amplitude and argmax index per trace with concurrency.
    Returns (max_vals, max_idxs) each shaped (R, T).
    - R: repeats
    - T: channels (n_traces)
    """
    R, T, L = traces.shape
    n_workers = min(int(max_workers), os.cpu_count() or 1)
    max_vals = np.empty((R, T), dtype=np.float64)
    max_idxs = np.empty((R, T), dtype=np.int64)

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
            mapped = exe.map(_max_amp_for_trace, gen, chunksize=4)
            if show_progress:
                mapped = tqdm(mapped, total=T, desc=f"Repeat {r+1}/{R}", unit="trace", leave=False)
            row_vals, row_idxs = zip(*mapped)
            max_vals[r, :] = np.asarray(row_vals, dtype=np.float64)
            max_idxs[r, :] = np.asarray(row_idxs, dtype=np.int64)

    return max_vals, max_idxs


# -------------------
# Main runner
# -------------------
def main() -> int:
    in_pattern = "/ceph/dwong/trigger_samples/injection_v2_300k/traces_energy_{energy}.zst"
    sub_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/sub_ch_template.npy")
    vac_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy")
    noise_psd   = np.load("/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy")

    fs        = 3906250
    repeats   = 100
    n_traces  = 56
    samples   = 300000
    energies  = list(range(0, 51, 2))  # eV
    out_dir   = Path("./indiv_max_amp")
    out_dir.mkdir(parents=True, exist_ok=True)

    for E in tqdm(energies, desc="Energies", unit="E"):
        in_path  = Path(in_pattern.format(energy=E))
        out_amp  = out_dir / f"max_amplitudes_energy_{E}.npy"
        out_idx  = out_dir / f"argmax_indices_energy_{E}.npy"

        if not in_path.exists():
            print(f"[skip] Input file not found: {in_path}")
            continue

        # Expect (repeats, n_traces, samples)
        traces = load_traces_from_zstd(in_path, n_traces=repeats, trace_shape=(n_traces, samples))
        if traces.shape != (repeats, n_traces, samples):
            raise ValueError(f"Bad shape {traces.shape}; expected {(repeats, n_traces, samples)}")

        max_vals, max_idxs = compute_max_amplitudes(
            traces=traces,
            sub_template=sub_template,
            vac_template=vac_template,
            noise_psd=noise_psd,
            fs=fs,
            max_workers=min(56, os.cpu_count() or 1),
            reanchor_every=None,
            show_progress=True,
        )

        np.save(out_amp, max_vals)
        np.save(out_idx, max_idxs)
        print(f"Saved {out_amp} with shape {max_vals.shape}")
        print(f"Saved {out_idx} with shape {max_idxs.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
