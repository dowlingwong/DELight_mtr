import yaml
import numpy as np
from TraceSimulator import LongTraceSimulator
from archive.trigger_study.archive.wk27.trace_IO import *
import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import math
import h5py
import os

# -------------------
# Load simulator config (one dict, reuseable)
# -------------------
def read_yaml_to_dict(file_path):
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict

CONFIG_PATH = "/home/dwong/DELight_mtr/trigger_study/wk24/training_config_small.yaml"
CONFIG = read_yaml_to_dict(CONFIG_PATH)

# -------------------
# Config
# -------------------
TOTAL_SETS   = 25000                     # total events per energy
BATCH_SIZE   = 1000                      # per batch
TRACE_SAMPLES = 65_536
DTYPE         = np.float16
COMPRESSION_LEVEL = 15

ENERGIES    = [10, 20, 50, 100, 200, 500, 1000]

# IMPORTANT: number of worker threads **per energy** (batches in parallel)
# Tune this to your CPU / I/O limits.
BATCH_WORKERS = 60

OUTDIR = Path("/ceph/dwong/work/training_samples/ER/small")

TYPE_RECOIL = "ER"
PHONON_ONLY = False
NO_NOISE    = False
QUANTIZE    = True
ROUTE_ALL_SIGNAL_TO_CH0 = False

# -------------------
# Streaming helpers
# -------------------
def _shuffle_bytes(arr: np.ndarray, dtype=DTYPE) -> bytes:
    a = np.asarray(arr, dtype=dtype)
    return a.view(np.uint8).reshape(-1, a.itemsize).T.tobytes()

def save_traces_streaming(traces_iter, output_path: Path, compression_level=COMPRESSION_LEVEL):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=compression_level)
    with open(output_path, "wb") as f:
        with cctx.stream_writer(f) as zw:
            for tr in traces_iter:
                zw.write(_shuffle_bytes(tr))

# -------------------
# Metadata helpers
# -------------------
def create_meta_h5(path: Path, count: int, n_channels: int, trace_samples: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    meta_dt = np.dtype([
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("energy", np.float64),
        ("type_recoil", str_dt),
        ("no_noise", np.bool_),
        ("quantize", np.bool_),
    ])
    f = h5py.File(path, "w")
    dset = f.create_dataset("events", shape=(count,), dtype=meta_dt, chunks=True)
    f.attrs["n_channels"] = int(n_channels)
    f.attrs["trace_samples"] = int(trace_samples)
    f.attrs["trace_dtype"] = np.dtype(DTYPE).name
    f.attrs["compression"] = f"zstd:{COMPRESSION_LEVEL}"
    return f, dset

# -------------------
# Generate one batch (thread-safe: instantiate LTS inside)
# -------------------
def generate_and_save_batch(energy: float, batch_idx: int, batch_size: int):
    """
    Generates `batch_size` events for `energy` into:
      - traces_energy_<E>_batch_<k>.zst
      - meta_energy_<E>_batch_<k>.h5
    """
    # Each worker builds its own simulator (avoids shared-state/thread issues)
    lts = LongTraceSimulator(CONFIG)

    traces_out = OUTDIR / f"traces_energy_{energy}_batch_{batch_idx:04d}.zst"
    meta_out   = OUTDIR / f"meta_energy_{energy}_batch_{batch_idx:04d}.h5"

    # Prime to discover shape
    traces, pos = lts.generate(
        E=energy, x=None, y=None, z=None,
        type_recoil=TYPE_RECOIL,
        phonon_only=PHONON_ONLY,
        no_noise=NO_NOISE,
        quantize=QUANTIZE,
        return_signal_mask=False,
        route_all_signal_to_ch0=ROUTE_ALL_SIGNAL_TO_CH0,
    )
    event_traces = np.asarray(traces[0], dtype=DTYPE)
    n_channels, n_samples = event_traces.shape
    if n_samples != TRACE_SAMPLES:
        raise ValueError(f"Trace length {n_samples} != {TRACE_SAMPLES}")

    h5f, meta_dset = create_meta_h5(meta_out, batch_size, n_channels, TRACE_SAMPLES)

    def _scalar(v): return float(np.ravel(v)[0])

    def traces_iter():
        # write primed first event
        x0, y0, z0 = pos
        meta_dset[0] = (_scalar(x0), _scalar(y0), _scalar(z0),
                        float(energy), TYPE_RECOIL, bool(NO_NOISE), bool(QUANTIZE))
        yield event_traces

        # remaining events
        for i in range(1, batch_size):
            ts, pos_i = lts.generate(
                E=energy, x=None, y=None, z=None,
                type_recoil=TYPE_RECOIL,
                phonon_only=PHONON_ONLY,
                no_noise=NO_NOISE,
                quantize=QUANTIZE,
                return_signal_mask=False,
                route_all_signal_to_ch0=ROUTE_ALL_SIGNAL_TO_CH0,
            )
            ev_tr = np.asarray(ts[0], dtype=DTYPE)
            x, y, z = pos_i
            meta_dset[i] = (_scalar(x), _scalar(y), _scalar(z),
                            float(energy), TYPE_RECOIL, bool(NO_NOISE), bool(QUANTIZE))
            yield ev_tr

    save_traces_streaming(traces_iter(), traces_out)
    h5f.flush()
    h5f.close()
    return n_channels

# -------------------
# Merge and cleanup
# -------------------
def merge_batches(energy: float, batch_sizes, n_channels: int):
    total_events = sum(batch_sizes)

    # merge traces
    merged_trace_path = OUTDIR / f"traces_energy_{energy}.zst"
    cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
    with open(merged_trace_path, "wb") as fout:
        with cctx.stream_writer(fout) as zw:
            for k in range(len(batch_sizes)):
                batch_file = OUTDIR / f"traces_energy_{energy}_batch_{k:04d}.zst"
                if batch_file.exists():
                    with open(batch_file, "rb") as f:
                        zw.write(f.read())
                    batch_file.unlink()

    # merge metadata
    merged_meta_path = OUTDIR / f"meta_energy_{energy}.h5"
    with h5py.File(merged_meta_path, "w") as fout:
        str_dt = h5py.string_dtype(encoding="utf-8")
        meta_dt = np.dtype([
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("energy", np.float64),
            ("type_recoil", str_dt),
            ("no_noise", np.bool_),
            ("quantize", np.bool_),
        ])
        dset = fout.create_dataset("events", shape=(total_events,), dtype=meta_dt, chunks=True)
        fout.attrs["n_channels"] = n_channels
        fout.attrs["trace_samples"] = TRACE_SAMPLES
        fout.attrs["trace_dtype"] = np.dtype(DTYPE).name
        fout.attrs["compression"] = f"zstd:{COMPRESSION_LEVEL}"

        idx = 0
        for k, bsz in enumerate(batch_sizes):
            meta_file = OUTDIR / f"meta_energy_{energy}_batch_{k:04d}.h5"
            if meta_file.exists():
                with h5py.File(meta_file, "r") as fin:
                    arr = fin["events"][:]
                    dset[idx:idx + len(arr)] = arr
                    idx += len(arr)
                meta_file.unlink()

# -------------------
# Run one energy: batches in parallel, then merge
# -------------------
def run_one_energy(energy: float):
    # figure out all batch sizes (handle short last batch)
    n_full, rem = divmod(TOTAL_SETS, BATCH_SIZE)
    batch_sizes = [BATCH_SIZE] * n_full + ([rem] if rem else [])
    if not batch_sizes:
        return

    n_channels = None
    with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as ex:
        futures = {}
        for k, bsz in enumerate(batch_sizes):
            futures[ex.submit(generate_and_save_batch, energy, k, bsz)] = (k, bsz)
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"E={energy} (batches)"):
            k, bsz = futures[fut]
            try:
                n_channels = fut.result()
            except Exception as exn:
                print(f"[energy={energy} batch={k}] FAILED: {exn}")
                raise

    # merge and clean up
    merge_batches(energy, batch_sizes, n_channels)

# -------------------
# Main: energies sequential
# -------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for e in ENERGIES:          # <-- sequential over energies
        run_one_energy(e)

if __name__ == "__main__":
    main()
