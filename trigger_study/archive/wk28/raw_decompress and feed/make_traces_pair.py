import yaml
import numpy as np
from TraceSimulator import LongTraceSimulator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import h5py

# -------------------
# Load simulator config (one dict, reusable)
# -------------------
def read_yaml_to_dict(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


CONFIG_PATH = "/home/dwong/DELight_mtr/trigger_study/archive/wk24/training_config_small.yaml"
CONFIG = read_yaml_to_dict(CONFIG_PATH)

# -------------------
# Config (mirrors make_traces.py; only one energy/type)
# -------------------
TOTAL_SETS = 25000
BATCH_SIZE = 100
TRACE_SAMPLES = 65536
DTYPE = np.float16

ENERGY = 200.0
RECOIL_TYPE = "NR"

# Keep worker count low, I/O bounded
BATCH_WORKERS = 7

OUTDIR = Path("/ceph/srv/dwong/training_samples_h5/wk28_pairs")

PHONON_ONLY = False
NO_NOISE = False
QUANTIZE = True
ROUTE_ALL_SIGNAL_TO_CH0 = False


def _meta_dtype():
    str_dt = h5py.string_dtype(encoding="utf-8")
    return np.dtype(
        [
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("energy", np.float64),
            ("type_recoil", str_dt),
            ("no_noise", np.bool_),
            ("quantize", np.bool_),
        ]
    )


# -------------------
# Metadata helpers
# -------------------
def create_batch_h5(path: Path, count: int, n_channels: int, trace_samples: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_dt = _meta_dtype()
    f = h5py.File(path, "w")
    meta_dset = f.create_dataset("events", shape=(count,), dtype=meta_dt, chunks=True)
    traces_dset = f.create_dataset(
        "traces",
        shape=(count, n_channels, trace_samples),
        dtype=DTYPE,
    )
    clean_dset = f.create_dataset(
        "traces_clean",
        shape=(count, n_channels, trace_samples),
        dtype=DTYPE,
    )
    return f, meta_dset, traces_dset, clean_dset


def _scalar(v):
    return float(np.ravel(v)[0])


def _generate_pair(lts: LongTraceSimulator):
    """
    Generate a noisy trace and a clean counterpart from a single simulator
    call with return_pair=True so the positions stay identical.
    """
    out = lts.generate(
        E=ENERGY,
        type_recoil=RECOIL_TYPE,
        phonon_only=PHONON_ONLY,
        no_noise=NO_NOISE,
        quantize=QUANTIZE,
        return_signal_mask=False,
        route_all_signal_to_ch0=ROUTE_ALL_SIGNAL_TO_CH0,
        return_pair=True,
    )

    pos = (None, None, None)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
        clean_ts, noisy_ts = out[0]
        if isinstance(out[1], tuple) and len(out[1]) == 3:
            pos = out[1]
    else:
        clean_ts, noisy_ts = out

    return noisy_ts, clean_ts, pos


# -------------------
# Generate one batch (thread-safe: instantiate LTS inside)
# -------------------
def generate_and_save_batch(batch_idx: int, batch_size: int, outdir: Path):
    """
    Generates `batch_size` paired events for fixed energy into:
      - NR_traces_energy_200_pair_batch_<k>.h5 (noisy + clean + metadata)
    """
    lts = LongTraceSimulator(CONFIG)

    batch_out = outdir / f"{RECOIL_TYPE}_traces_energy_{int(ENERGY)}_pair_batch_{batch_idx:04d}.h5"

    traces, traces_clean, pos = _generate_pair(lts)
    event_traces = np.asarray(traces[0], dtype=DTYPE)
    event_traces_clean = np.asarray(traces_clean[0], dtype=DTYPE)
    n_channels, n_samples = event_traces.shape
    if n_samples != TRACE_SAMPLES:
        raise ValueError(f"Trace length {n_samples} != {TRACE_SAMPLES}")

    h5f, meta_dset, trace_dset, clean_dset = create_batch_h5(batch_out, batch_size, n_channels, TRACE_SAMPLES)

    x0, y0, z0 = pos
    meta_dset[0] = (
        _scalar(x0) if x0 is not None else np.nan,
        _scalar(y0) if y0 is not None else np.nan,
        _scalar(z0) if z0 is not None else np.nan,
        float(ENERGY),
        RECOIL_TYPE,
        bool(NO_NOISE),
        bool(QUANTIZE),
    )
    trace_dset[0] = event_traces
    clean_dset[0] = event_traces_clean
    del event_traces, traces

    for i in range(1, batch_size):
        ts_noisy, ts_clean, pos_i = _generate_pair(lts)
        ev_tr = np.asarray(ts_noisy[0], dtype=DTYPE)
        ev_clean = np.asarray(ts_clean[0], dtype=DTYPE)
        x, y, z = pos_i
        meta_dset[i] = (
            _scalar(x) if x is not None else np.nan,
            _scalar(y) if y is not None else np.nan,
            _scalar(z) if z is not None else np.nan,
            float(ENERGY),
            RECOIL_TYPE,
            bool(NO_NOISE),
            bool(QUANTIZE),
        )
        trace_dset[i] = ev_tr
        clean_dset[i] = ev_clean
        del ev_tr, ts_noisy

    h5f.flush()
    h5f.close()
    return n_channels


# -------------------
# Run fixed energy: batches in parallel
# -------------------
def run_one_energy(outdir: Path):
    n_full, rem = divmod(TOTAL_SETS, BATCH_SIZE)
    batch_sizes = [BATCH_SIZE] * n_full + ([rem] if rem else [])
    if not batch_sizes:
        return

    with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as ex:
        futures = {}
        for k, bsz in enumerate(batch_sizes):
            futures[ex.submit(generate_and_save_batch, k, bsz, outdir)] = (k, bsz)
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{RECOIL_TYPE} E={ENERGY} pair (batches)"):
            k, bsz = futures[fut]
            try:
                fut.result()
            except Exception as exn:
                print(f"[type={RECOIL_TYPE} energy={ENERGY} batch={k}] FAILED: {exn}")
                raise


# -------------------
# Main: single energy/type
# -------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    run_one_energy(OUTDIR)


if __name__ == "__main__":
    main()
