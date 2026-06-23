import yaml
import numpy as np
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import h5py

from TraceSimulator import LongTraceSimulator
from archive.trigger_study.archive.wk26.trace_IO import *  # if you actually need things from here


# -------------------
# Paths & globals
# -------------------
CONFIG_PATH = "/home/dwong/DELight_mtr/trigger_study/wk26/training_config.yaml"
BASE_OUTDIR = Path("/ceph/dwong/work/threshold")  # base folder

# Noise powers to scan
NOISE_POWERS = [13.2, 17.2, 21.2, 25.2, 29.2,
                33.2, 37.2, 41.2, 45.2, 49.2, 53.2]

# -------------------
# Load simulator config
# -------------------
def read_yaml_to_dict(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# -------------------
# Config
# -------------------
TOTAL_SETS       = 60        # events per energy
TRACE_SAMPLES    = 50_000
DTYPE            = np.float16
COMPRESSION_LEVEL = 15

ENERGIES         = list(range(0, 61, 2))  # 0,2,...,58 (30 energies)

# Use 26 cores: each core handles one energy at a time
ENERGY_WORKERS   = 7

TYPE_RECOIL      = "NR"
PHONON_ONLY      = False
NO_NOISE         = False
QUANTIZE         = True
ROUTE_ALL_SIGNAL_TO_CH0 = False


# -------------------
# Streaming helpers
# -------------------
def _shuffle_bytes(arr: np.ndarray, dtype=DTYPE) -> bytes:
    """Channel-major byte shuffle for better compression."""
    a = np.asarray(arr, dtype=dtype)
    return a.view(np.uint8).reshape(-1, a.itemsize).T.tobytes()


def save_traces_streaming(traces_iter, output_path: Path, compression_level=COMPRESSION_LEVEL):
    """
    Write a single zstd stream containing all events from traces_iter.
    Each trace in traces_iter should be shape (n_channels, TRACE_SAMPLES).
    """
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
    """Create metadata HDF5 for `count` events, return (file, dataset)."""
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
# Worker: generate one energy (vectorized over 200 events)
# -------------------
def generate_one_energy(energy: float, config: dict, outdir: Path):
    """
    Generate exactly TOTAL_SETS events for this energy into:
      - traces_energy_<E>.zst
      - meta_energy_<E>.h5

    Uses vectorized generate: E is a length-TOTAL_SETS list.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    traces_out = outdir / f"traces_energy_{energy}.zst"
    meta_out   = outdir / f"meta_energy_{energy}.h5"

    # Build simulator in this process
    lts = LongTraceSimulator(config)

    # Vectorized energy: 200 events of same energy
    E_vec = [energy] * TOTAL_SETS

    # NOTE: keep extra args if your LongTraceSimulator.generate supports them
    ts, (x, y, z) = lts.generate(
        E=E_vec,
        x=None, y=None, z=None,
        type_recoil=TYPE_RECOIL,
        phonon_only=PHONON_ONLY,
        no_noise=NO_NOISE,
        quantize=QUANTIZE,
        return_signal_mask=False,
        route_all_signal_to_ch0=ROUTE_ALL_SIGNAL_TO_CH0,
    )

    # ts shape is (TOTAL_SETS, n_channels, TRACE_SAMPLES)
    if ts.shape[0] != TOTAL_SETS:
        raise ValueError(f"[E={energy}] Got {ts.shape[0]} events, expected {TOTAL_SETS}")
    n_events, n_channels, n_samples = ts.shape
    if n_samples != TRACE_SAMPLES:
        raise ValueError(f"[E={energy}] Trace length {n_samples} != {TRACE_SAMPLES}")

    # Create metadata for all events
    h5f, meta_dset = create_meta_h5(meta_out, n_events, n_channels, TRACE_SAMPLES)

    def _scalar(v):
        return float(np.ravel(v)[0])

    def traces_iter():
        # write metadata + traces for all events
        for i in range(n_events):
            meta_dset[i] = (_scalar(x[i]), _scalar(y[i]), _scalar(z[i]),
                            float(energy), TYPE_RECOIL, bool(NO_NOISE), bool(QUANTIZE))
            yield ts[i]

    # Stream all events into one zstd file
    save_traces_streaming(traces_iter(), traces_out)
    h5f.flush()
    h5f.close()

    print(f"[worker] {outdir.name} | E={energy}: wrote {n_events} events")
    return energy


def _worker_generate_one_energy(args):
    energy, config, outdir = args
    return generate_one_energy(energy, config, outdir)


# -------------------
# Main: loop over noise_power values, energies in parallel
# -------------------
def main():
    for npw in NOISE_POWERS:
        print(f"\n=== Running with noise_power = {npw} ===")

        # Load base config and set noise_power directly in the dict
        cfg = read_yaml_to_dict(CONFIG_PATH)
        cfg["noise_power"] = float(npw)

        # Optional: persist this noise_power to YAML for record-keeping
        # with open(CONFIG_PATH, "w") as f:
        #     yaml.safe_dump(cfg, f, sort_keys=False)

        outdir = BASE_OUTDIR / f"noise_{npw:.1f}"
        outdir.mkdir(parents=True, exist_ok=True)
        print("Saving traces to:", outdir)

        tasks = [(e, cfg, outdir) for e in ENERGIES]

        # Use 26 processes, each doing one energy at a time
        with ProcessPoolExecutor(max_workers=ENERGY_WORKERS) as ex:
            list(tqdm(
                ex.map(_worker_generate_one_energy, tasks),
                total=len(tasks),
                desc=f"noise={npw:.1f} | energies",
                unit="E"
            ))

        print(f"=== Finished noise_power = {npw} ===")

if __name__ == "__main__":
    main()