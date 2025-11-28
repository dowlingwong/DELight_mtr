import yaml
import numpy as np
from TraceSimulator import LongTraceSimulator
from wk27.trace_IO import *
import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import math
import h5py
import os

# -------------------
# Load simulator
# -------------------
def read_yaml_to_dict(file_path):
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict

config = read_yaml_to_dict("/home/dwong/DELight_mtr/trigger_study/wk24/training_config.yaml")
lts = LongTraceSimulator(config)

# -------------------
# Config
# -------------------
TOTAL_SETS = 2500              # total events per energy
BATCH_SIZE = 100               # per batch
TRACE_SAMPLES = 150_000
DTYPE = np.float16
COMPRESSION_LEVEL = 15
ENERGIES = [10, 20, 50, 100, 200, 500, 1000]
MAX_THREADS = 50
OUTDIR = Path("/ceph/dwong/work/training_samples/NR")

TYPE_RECOIL = "NR"
PHONON_ONLY = False
NO_NOISE = False
QUANTIZE = True
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

def load_traces_from_zstd(input_path: Path, n_traces: int, dtype=DTYPE, trace_shape=None) -> np.ndarray:
    if trace_shape is None:
        raise ValueError("trace_shape must be provided to load traces.")
    def _unshuffle_bytes(data: bytes, dtype=dtype, shape=trace_shape) -> np.ndarray:
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
# Generate one batch
# -------------------
def generate_and_save_batch(energy: float, batch_idx: int, batch_size: int):
    traces_out = OUTDIR / f"traces_energy_{energy}_batch_{batch_idx:04d}.zst"
    meta_out = OUTDIR / f"meta_energy_{energy}_batch_{batch_idx:04d}.h5"

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
        for i in range(batch_size):
            ts, pos = lts.generate(
                E=energy, x=None, y=None, z=None,
                type_recoil=TYPE_RECOIL,
                phonon_only=PHONON_ONLY,
                no_noise=NO_NOISE,
                quantize=QUANTIZE,
                return_signal_mask=False,
                route_all_signal_to_ch0=ROUTE_ALL_SIGNAL_TO_CH0,
            )
            ev_tr = np.asarray(ts[0], dtype=DTYPE)
            x, y, z = pos
            meta_dset[i] = (_scalar(x), _scalar(y), _scalar(z),
                            float(energy), TYPE_RECOIL,
                            bool(NO_NOISE), bool(QUANTIZE))
            yield ev_tr

    save_traces_streaming(traces_iter(), traces_out)
    h5f.flush()
    h5f.close()
    return n_channels

# -------------------
# Merge and cleanup
# -------------------
def merge_batches(energy: float, n_batches: int, total_events: int, n_channels: int):
    # ---- merge traces ----
    merged_trace_path = OUTDIR / f"traces_energy_{energy}.zst"
    cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
    with open(merged_trace_path, "wb") as fout:
        with cctx.stream_writer(fout) as zw:
            for k in range(n_batches):
                batch_file = OUTDIR / f"traces_energy_{energy}_batch_{k:04d}.zst"
                if not batch_file.exists():
                    continue
                with open(batch_file, "rb") as f:
                    zw.write(f.read())
                batch_file.unlink()  # delete after merging

    # ---- merge metadata ----
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
        for k in range(n_batches):
            meta_file = OUTDIR / f"meta_energy_{energy}_batch_{k:04d}.h5"
            if not meta_file.exists():
                continue
            with h5py.File(meta_file, "r") as fin:
                arr = fin["events"][:]
                dset[idx:idx + len(arr)] = arr
                idx += len(arr)
            meta_file.unlink()  # delete small file

# -------------------
# Generate per energy
# -------------------
def generate_energy_in_batches(energy: float, total_sets: int, batch_size: int):
    n_batches = math.ceil(total_sets / batch_size)
    n_channels = None
    for k in tqdm(range(n_batches), desc=f"E={energy}"):
        n_channels = generate_and_save_batch(energy, k, batch_size)
    merge_batches(energy, n_batches, total_sets, n_channels)

# -------------------
# Main driver
# -------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        futures = {ex.submit(generate_energy_in_batches, e, TOTAL_SETS, BATCH_SIZE): e for e in ENERGIES}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Energies"):
            e = futures[fut]
            try:
                fut.result()
            except Exception as exn:
                print(f"[energy={e}] FAILED: {exn}")
                raise

if __name__ == "__main__":
    main()
