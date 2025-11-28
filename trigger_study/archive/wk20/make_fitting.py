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
from multiprocessing import shared_memory  # NEW

from wk27.OptimumFilter import OptimumFilter

# -------------------
# Worker process globals
# -------------------
_WORKER_OF_SUB: OptimumFilter | None = None
_WORKER_OF_VAC: OptimumFilter | None = None
_REANCHOR_EVERY: int | None = None
_FIRST_SUB_CHANNELS = 19  # first 19 channels use submerged template

# Shared memory handles (set in _worker_init)
_TRACES_SHM = None
_TRACES_SHAPE = None
_TRACES_DTYPE = None
_TRACES_NP = None  # np.ndarray view backed by shared memory


def _worker_init(
    sub_template: np.ndarray,
    vac_template: np.ndarray,
    noise_psd: np.ndarray,
    fs: float,
    reanchor_every: int | None,
    shm_name: str | None,
    traces_shape: tuple[int, int, int] | None,
    traces_dtype: str | None,
):
    """Initializer runs once per worker. Builds two OptimumFilters and attaches shared memory."""
    # keep worker threads single-threaded to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    global _WORKER_OF_SUB, _WORKER_OF_VAC, _REANCHOR_EVERY
    _WORKER_OF_SUB = OptimumFilter(sub_template, noise_psd, fs)
    _WORKER_OF_VAC = OptimumFilter(vac_template, noise_psd, fs)
    _REANCHOR_EVERY = reanchor_every

    # attach to shared memory
    global _TRACES_SHM, _TRACES_SHAPE, _TRACES_DTYPE, _TRACES_NP
    if shm_name is not None:
        _TRACES_SHM = shared_memory.SharedMemory(name=shm_name)
        _TRACES_SHAPE = tuple(traces_shape)
        _TRACES_DTYPE = np.dtype(traces_dtype)
        _TRACES_NP = np.ndarray(_TRACES_SHAPE, dtype=_TRACES_DTYPE, buffer=_TRACES_SHM.buf)


def _max_amp_for_task(args) -> tuple[int, int, float, int]:
    """
    Compute max amplitude for one (repeat r, channel ch).
    Returns (r, ch, max_amp, argmax_idx).
    """
    r, ch = args
    x = np.ascontiguousarray(_TRACES_NP[r, ch, :], dtype=np.float64)

    # choose filter by channel index
    if ch < _FIRST_SUB_CHANNELS:
        amps, _ = _WORKER_OF_SUB.sliding_fit(x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none")
    else:
        amps, _ = _WORKER_OF_VAC.sliding_fit(x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none")

    amps = np.asarray(amps, dtype=np.float64)
    idx = int(np.argmax(amps))
    return r, ch, float(amps[idx]), idx


# -------------------
# Loader for traces from Zstandard
# -------------------
def load_traces_from_zstd(input_path: Path, n_traces: int, trace_shape: tuple[int, ...],
                          dtype=np.float16) -> np.ndarray:
    """Load stacked ndarray of shape (n_traces, *trace_shape) from a .zst file."""
    if trace_shape is None:
        raise ValueError("trace_shape must be provided.")

    def _unshuffle_bytes(data: bytes, dtype, shape):
        # inverse of bytewise shuffle; safe fallback if used during save
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
# Core computation (FASTER + honors explicit worker count)
# -------------------
def compute_max_amplitudes(
    traces: np.ndarray,
    sub_template: np.ndarray,
    vac_template: np.ndarray,
    noise_psd: np.ndarray,
    fs: float,
    *,
    max_workers: int | None = None,   # EXACT number of workers if provided
    reanchor_every: int | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute max OF amplitude and argmax index per trace with concurrency.
    Returns (max_vals, max_idxs) each shaped (R, T).
    """
    R, T, L = traces.shape

    # If user provides max_workers, use it EXACTLY (no implicit cap).
    # If None, default to all visible CPUs.
    if max_workers is None:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = int(max_workers)
        if n_workers < 1:
            raise ValueError("max_workers must be >= 1")

    # Move 'traces' into shared memory to avoid pickling costs
    shm = shared_memory.SharedMemory(create=True, size=traces.nbytes)
    shm_np = np.ndarray(traces.shape, dtype=traces.dtype, buffer=shm.buf)
    shm_np[...] = traces  # one-time copy

    max_vals = np.empty((R, T), dtype=np.float64)
    max_idxs = np.empty((R, T), dtype=np.int64)

    # build all tasks once
    tasks = ((r, ch) for r in range(R) for ch in range(T))
    total_tasks = R * T

    # tune chunksize to reduce scheduler overhead
    # rule of thumb: ~8 chunks per worker, but not tiny
    chunksize = max(8, total_tasks // max(n_workers * 8, 1))

    if show_progress:
        print(f"[compute_max_amplitudes] requested_workers={n_workers}, visible_cpus={os.cpu_count() or 1}, "
              f"R={R}, T={T}, L={L}, tasks={total_tasks}, chunksize={chunksize}")

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(
                sub_template, vac_template, noise_psd, fs, reanchor_every,
                shm.name, traces.shape, str(traces.dtype),
            ),
        ) as exe:
            iterator = exe.map(_max_amp_for_task, tasks, chunksize=chunksize)
            if show_progress:
                iterator = tqdm(iterator, total=total_tasks, desc="Fitting", unit="job")
            for r, ch, val, idx in iterator:
                max_vals[r, ch] = val
                max_idxs[r, ch] = idx
    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()

    return max_vals, max_idxs


# -------------------
# Main runner
# -------------------
def main() -> int:
    in_pattern = "/ceph/dwong/trigger_samples/400sample_v2_300k/traces_energy_{energy}.zst"
    sub_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/sub_ch_template.npy")
    vac_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy")
    noise_psd   = np.load("/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy")

    fs        = 3906250
    repeats   = 400
    n_traces  = 56
    samples   = 300000
    energies  = list(range(34, 50, 2))
    out_dir   = Path("./max_amp_400")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ENV guard against threaded BLAS inside workers (kept here as well)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # *** Set EXACT worker count here. No hidden cap. ***
    requested_workers = 56

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
            max_workers=requested_workers,   # uses EXACTLY 56
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
