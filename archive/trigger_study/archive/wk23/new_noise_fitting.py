#!/usr/bin/env python3
"""
Single-file pipeline with baked-in defaults (no CLI args).

- For each listed noise PSD, update the YAML config's 'noise_psd' field to the *generation* PSD path
- Generate traces per energy (samples/config are kept the same as your YAML; we do not change other config fields)
- Fit max OF amplitudes using the SAME templates as before, but with the CORRESPONDING *filter* PSD

Outputs:
<BASE_OUT>/<noise_name>/traces/traces_energy_{E}.zst
<BASE_OUT>/<noise_name>/amplitudes/max_amplitudes_energy_{E}.npy
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import numpy as np
import zstandard as zstd
from tqdm.auto import tqdm

# --- External deps expected in your environment ---
# - TraceSimulator.LongTraceSimulator
# - OptimumFilter.OptimumFilter
from TraceSimulator import LongTraceSimulator
from archive.trigger_study.archive.wk27.OptimumFilter import OptimumFilter

# ===================
# DEFAULTS (edit here if needed)
# ===================
CONFIG_YAML   = Path("/ceph/dwong/delight_conf/ETP_config_v2.yaml")
BASE_OUT      = Path("/ceph/dwong/trigger_samples")
SUB_TEMPLATE  = Path("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/sub_ch_template.npy")
VAC_TEMPLATE  = Path("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy")
FS_HZ         = 3_906_250.0
ENERGIES      = list(range(0, 27, 2))
N_SETS        = 50        # repeats per energy
N_TRACES      = 56         # channels per trace
SAMPLES       = 300_000    # trace samples
MAX_THREADS   = 30         # generation threads
MAX_WORKERS   = 56         # fitting processes
REANCHOR_EVERY = None      # e.g. 512 if you want periodic re-anchoring

# ===================
# NEW: separate PSD directories for generation vs. OF fitting
# ===================
GEN_NOISE_BASE    = Path("/home/dwong/DELight_mtr/trigger_study/wk23/new_noise_psd_generate/")
FILTER_NOISE_BASE = Path("/home/dwong/DELight_mtr/trigger_study/wk23/new_noise_psd_filter/")

# Use the same noise names across phases; filenames are "{name}.npy" in each base dir
NOISE_NAMES = [
    "Johnson_noise",
    "Er_noise",
    "TD_noise",
    "total_noise",
    "SQUID_noise",
]

def gen_noise_path(name: str) -> Path:
    return GEN_NOISE_BASE / f"{name}.npy"

def filter_noise_path(name: str) -> Path:
    return FILTER_NOISE_BASE / f"{name}.npy"

# -------------------
# Byte-shuffle helpers for better compression
# -------------------
DTYPE = np.float16
COMPRESSION_LEVEL = 15

def _shuffle_bytes(arr: np.ndarray) -> bytes:
    return arr.view(np.uint8).reshape(-1, arr.itemsize).T.tobytes()

def _unshuffle_bytes(data: bytes, dtype, shape):
    itemsize = np.dtype(dtype).itemsize
    num_elements = int(np.prod(shape))
    reshaped = np.frombuffer(data, dtype=np.uint8).reshape(itemsize, num_elements).T
    unshuffled = reshaped.reshape(-1)
    return unshuffled.view(dtype).reshape(shape)

def save_traces_to_zstd(traces_list, output_path: Path,
                        dtype=DTYPE, trace_shape=None, compression_level=COMPRESSION_LEVEL):
    if trace_shape is None:
        if not traces_list:
            raise ValueError("No traces to save and trace_shape not provided.")
        trace_shape = traces_list[0].shape

    all_data = bytearray()
    for trace in traces_list:
        if trace.shape != trace_shape:
            raise ValueError(f"Trace has wrong shape {trace.shape}, expected {trace_shape}")
        shuffled = _shuffle_bytes(np.asarray(trace, dtype=dtype))
        all_data.extend(shuffled)

    compressor = zstd.ZstdCompressor(level=compression_level)
    compressed_data = compressor.compress(bytes(all_data))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(compressed_data)

def load_traces_from_zstd(input_path: Path, n_traces: int, trace_shape: tuple[int, ...],
                          dtype=np.float16) -> np.ndarray:
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
# Generation
# -------------------
def read_yaml_to_dict(file_path: Path) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def write_yaml_from_dict(file_path: Path, data: dict) -> None:
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)

def build_lts_from_config(config_path: Path) -> LongTraceSimulator:
    config = read_yaml_to_dict(config_path)
    return LongTraceSimulator(config)

def generate_one_energy(lts, energy: int, n_sets: int, trace_samples: int, outdir: Path, dtype=DTYPE):
    all_traces = []
    all_masks = []
    for _ in range(n_sets):
        traces, signal_mask = lts.generate(
            E=energy,
            x=0, y=0, z=-1700,
            type_recoil='NR',
            phonon_only=False,
            no_noise=False,
            quantize=True,
            return_signal_mask=True,
        )
        # Expected shapes: traces -> (1, n_channels, trace_samples), mask -> (1, n_channels)
        event_traces = np.asarray(traces[0], dtype=dtype)      # (n_channels, trace_samples)
        event_mask = np.asarray(signal_mask[0], dtype=bool)    # (n_channels,)
        if event_traces.shape[1] != trace_samples:
            raise ValueError(f"Trace sample length mismatch: got {event_traces.shape[1]}, expected {trace_samples}")
        all_traces.append(event_traces)
        all_masks.append(event_mask)

    n_channels = all_traces[0].shape[0]
    trace_shape = (n_channels, trace_samples)
    traces_out_path = outdir / f"traces_energy_{energy}.zst"
    save_traces_to_zstd(all_traces, traces_out_path, dtype=dtype, trace_shape=trace_shape, compression_level=COMPRESSION_LEVEL)
    return traces_out_path

def run_generation(config_path: Path, energies, n_sets: int, trace_samples: int, outdir: Path, max_threads: int = 20):
    outdir.mkdir(parents=True, exist_ok=True)
    lts = build_lts_from_config(config_path)
    futures = []
    with ThreadPoolExecutor(max_workers=max_threads) as ex:
        for e in energies:
            futures.append(ex.submit(generate_one_energy, lts, e, n_sets, trace_samples, outdir))
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating Traces"):
            _ = fut.result()

# -------------------
# Fitting
# -------------------
_WORKER_OF_SUB = None
_WORKER_OF_VAC = None
_REANCHOR_EVERY = None

def _worker_init(sub_template: np.ndarray, vac_template: np.ndarray,
                 noise_psd: np.ndarray, fs: float, reanchor_every: int | None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    global _WORKER_OF_SUB, _WORKER_OF_VAC, _REANCHOR_EVERY
    _WORKER_OF_VAC = OptimumFilter(vac_template, noise_psd, fs)
    _WORKER_OF_SUB = OptimumFilter(sub_template, noise_psd, fs)
    _REANCHOR_EVERY = reanchor_every

def _max_amp_for_trace(args) -> float:
    trace_1d, ch = args
    x = np.ascontiguousarray(trace_1d, dtype=np.float64)
    if ch < 19:
        amps, _ = _WORKER_OF_SUB.sliding_fit(x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none")
    else:
        amps, _ = _WORKER_OF_VAC.sliding_fit(x, hop=1, reanchor_every=_REANCHOR_EVERY, chisq_mode="none")
    return float(np.max(amps))

def compute_max_amplitudes(traces: np.ndarray, sub_template: np.ndarray, vac_template: np.ndarray,
                           noise_psd: np.ndarray, fs: float, *, max_workers: int = 56,
                           reanchor_every: int | None = None, show_progress: bool = True) -> np.ndarray:
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
# Driver
# -------------------
def main():
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    # Load templates once (same as before)
    sub_template = np.load(SUB_TEMPLATE)
    vac_template = np.load(VAC_TEMPLATE)

    for noise_name in NOISE_NAMES:
        gen_psd_path = gen_noise_path(noise_name)
        filt_psd_path = filter_noise_path(noise_name)

        print(f"\n==== Noise: {noise_name} ====")
        # Basic existence checks (optional but helpful)
        if not gen_psd_path.exists():
            raise FileNotFoundError(f"Generation PSD not found: {gen_psd_path}")
        if not filt_psd_path.exists():
            raise FileNotFoundError(f"Filter PSD not found: {filt_psd_path}")

        # 1) Write a temp config with the SAME settings as base, only noise_psd swapped (to *generation* PSD)
        cfg_dict = read_yaml_to_dict(CONFIG_YAML)
        cfg_dict['noise_psd'] = str(gen_psd_path)
        tmp_cfg = BASE_OUT / f"tmp_config_{noise_name}.yaml"
        write_yaml_from_dict(tmp_cfg, cfg_dict)

        # 2) Prepare directories
        root = BASE_OUT / noise_name
        traces_dir = root / "traces"
        amps_dir = root / "amplitudes"
        traces_dir.mkdir(parents=True, exist_ok=True)
        amps_dir.mkdir(parents=True, exist_ok=True)

        # 3) Generate traces (we DO NOT modify other config fields)
        run_generation(
            config_path=tmp_cfg,
            energies=ENERGIES,
            n_sets=N_SETS,
            trace_samples=SAMPLES,
            outdir=traces_dir,
            max_threads=MAX_THREADS,
        )

        # 4) Fit per energy, using the corresponding *filter* PSD for OF registration
        noise_psd = np.load(filt_psd_path)
        for E in tqdm(ENERGIES, desc=f"Fitting [{noise_name}]", unit="E"):
            in_path = traces_dir / f"traces_energy_{E}.zst"
            if not in_path.exists():
                print(f"[skip] missing: {in_path}")
                continue
            traces = load_traces_from_zstd(in_path, n_traces=N_SETS, trace_shape=(N_TRACES, SAMPLES))
            max_vals = compute_max_amplitudes(
                traces=traces,
                sub_template=sub_template,
                vac_template=vac_template,
                noise_psd=noise_psd,
                fs=FS_HZ,
                max_workers=MAX_WORKERS,
                reanchor_every=REANCHOR_EVERY,
                show_progress=True,
            )
            out_path = amps_dir / f"max_amplitudes_energy_{E}.npy"
            np.save(out_path, max_vals)
            print(f"Saved {out_path} with shape {max_vals.shape}")

        print(f"[DONE] {noise_name}: output at {root}")

if __name__ == "__main__":
    main()
