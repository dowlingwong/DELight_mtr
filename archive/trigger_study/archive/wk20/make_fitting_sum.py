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
_OF_VAC = None          # single-channel vacuum OF
_SUM_OF_VAC = None      # OF for summed vacuum trace (PSD scaled by #channels)
_REANCHOR_EVERY = None
_HOP = None
_VAC_IDX = None         # slice/indexes of vacuum channels
_TEMPLATE_LEN = None    # (optional) if you need it later

def _worker_init(vac_template: np.ndarray, noise_psd: np.ndarray,
                 fs: float, reanchor_every: int | None, hop: int, vac_idx: tuple[int, int]):
    """Initializer runs once per worker. Builds two OptimumFilters for vacuum channels."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    global _OF_VAC, _SUM_OF_VAC, _REANCHOR_EVERY, _HOP, _VAC_IDX, _TEMPLATE_LEN
    _OF_VAC = OptimumFilter(vac_template, noise_psd, fs)

    n_vac = vac_idx[1] - vac_idx[0]
    sum_psd = noise_psd * n_vac  # assume independent noise when summing channels
    _SUM_OF_VAC = OptimumFilter(vac_template, sum_psd, fs)

    _REANCHOR_EVERY = reanchor_every
    _HOP = hop
    _VAC_IDX = vac_idx
    _TEMPLATE_LEN = len(vac_template)


def _compute_metrics_for_repeat(traces_for_repeat: np.ndarray) -> tuple[float, float]:
    """
    traces_for_repeat: (T, L) array for ONE repeat.
    Returns:
        (sum_of_filtered_max, filtered_sum_max)
    """
    # select vacuum channels (last 37 channels)
    start, end = _VAC_IDX
    vac_traces = traces_for_repeat[start:end, :]  # shape (37, L)

    # ---- Metric 1: sum of filtered per-channel outputs, then max ----
    amps_sum = None
    for k in range(vac_traces.shape[0]):
        x = np.ascontiguousarray(vac_traces[k, :], dtype=np.float64)
        amps, _ = _OF_VAC.sliding_fit(
            x, hop=_HOP, reanchor_every=_REANCHOR_EVERY, chisq_mode="none"
        )
        if amps_sum is None:
            amps_sum = amps
        else:
            # accumulate sample-wise sum of filtered outputs
            amps_sum += amps
    sum_of_filtered_max = float(np.max(amps_sum)) if amps_sum is not None else float("nan")

    # ---- Metric 2: sum the raw channels first, then filter once, then max ----
    x_sum = np.sum(vac_traces, axis=0)  # shape (L,)
    x_sum = np.ascontiguousarray(x_sum, dtype=np.float64)
    amps_sumtrace, _ = _SUM_OF_VAC.sliding_fit(
        x_sum, hop=_HOP, reanchor_every=_REANCHOR_EVERY, chisq_mode="none"
    )
    filtered_sum_max = float(np.max(amps_sumtrace))

    return (sum_of_filtered_max, filtered_sum_max)


# -------------------
# Loader for traces from Zstandard
# -------------------
def load_traces_from_zstd(input_path: Path, n_traces: int, trace_shape: tuple[int, ...],
                          dtype=np.float16) -> np.ndarray:
    """Load stacked ndarray of shape (n_traces, *trace_shape) from a .zst file."""
    if trace_shape is None:
        raise ValueError("trace_shape must be provided.")

    def _unshuffle_bytes(data: bytes, dtype, shape):
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

    return np.stack(traces).astype(np.float32)


# -------------------
# Core computation (now per-repeat, not per-channel)
# -------------------
def compute_vacuum_metrics(traces: np.ndarray, vac_template: np.ndarray,
                           noise_psd: np.ndarray, fs: float, *,
                           max_workers: int = 58,
                           reanchor_every: int | None = None,
                           hop: int = 1,
                           show_progress: bool = True) -> np.ndarray:
    """
    traces: (R, T, L) array. T must be >= 56, vacuum channels are [19:56).
    Returns: (R, 2) array: columns [sum_of_filtered_max, filtered_sum_max]
    """
    R, T, L = traces.shape
    if T < 56:
        raise ValueError(f"Expected at least 56 channels; got {T}")
    n_workers = min(int(max_workers), os.cpu_count() or 1)

    # indices for vacuum channels (0-based): 19..55 inclusive
    vac_idx = (19, 56)

    out = np.empty((R, 2), dtype=np.float64)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(vac_template, noise_psd, fs, reanchor_every, hop, vac_idx),
    ) as exe:
        r_iter = range(R)
        if show_progress:
            r_iter = tqdm(r_iter, desc="Repeats", unit="repeat")

        # map each repeat's (T, L) slice to the worker
        mapped = exe.map(_compute_metrics_for_repeat,
                         (traces[r, :, :] for r in r_iter),
                         chunksize=1)

        # if showing progress, wrap the generator with tqdm for total R
        if show_progress:
            mapped = tqdm(mapped, total=R, desc="Processing", unit="rep", leave=False)

        for i, (m1, m2) in enumerate(mapped):
            out[i, 0] = m1
            out[i, 1] = m2

    return out


# -------------------
# Main runner
# -------------------
def main() -> int:
    in_pattern = "/ceph/dwong/trigger_samples/400sample_v2_300k/traces_energy_{energy}.zst"

    # Only vacuum template/PSD needed
    vac_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy")
    noise_psd = np.load("/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy")

    fs = 3906250
    repeats = 400
    n_traces = 56
    samples = 300000
    energies = list(range(0, 50, 2))
    out_dir = Path("./max_amp_pt")
    out_dir.mkdir(parents=True, exist_ok=True)

    for E in tqdm(energies, desc="Energies", unit="E"):
        in_path = Path(in_pattern.format(energy=E))
        out_path = out_dir / f"vacuum_metrics_energy_{E}.npy"  # shape (repeats, 2)

        if not in_path.exists():
            print(f"[skip] Input file not found: {in_path}")
            continue

        # traces shape: (repeats, 56, samples)
        traces = load_traces_from_zstd(in_path, n_traces=repeats, trace_shape=(n_traces, samples))
        if traces.shape != (repeats, n_traces, samples):
            raise ValueError(f"Bad shape {traces.shape}")

        metrics = compute_vacuum_metrics(
            traces=traces,
            vac_template=vac_template,
            noise_psd=noise_psd,
            fs=fs,
            max_workers=58,
            reanchor_every=None,
            hop=1,
            show_progress=True,
        )
        np.save(out_path, metrics)
        print(f"Saved {out_path} with shape {metrics.shape} (cols=[sum_of_filtered_max, filtered_sum_max])")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
