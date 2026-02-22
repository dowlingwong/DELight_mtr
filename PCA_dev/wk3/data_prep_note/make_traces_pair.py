import yaml
import numpy as np
from TraceSimulator import LongTraceSimulator
from TraceSimulator import NoiseGenerator
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


CONFIG_PATH = "/home/dwong/DELight_mtr/PCA_dev/reusable/PCA_config.yaml"
CONFIG = read_yaml_to_dict(CONFIG_PATH)

# -------------------
# Config (mirrors make_traces.py; only one energy/type)
# -------------------
TOTAL_SETS = 2000
BATCH_SIZE = 100
TRACE_SAMPLES = 32768
DTYPE = np.float32

ENERGIES = [500]
ENERGY = None
RECOIL_TYPE = "NR"

# Keep worker count low, I/O bounded
BATCH_WORKERS = 7

OUTDIR = Path("/ceph/dwong/trigger_samples/PCA_QP/main")

PHONON_ONLY = True
NO_NOISE = False
QUANTIZE = False
ROUTE_ALL_SIGNAL_TO_CH0 = False
QP_CHANNEL_MIN_INDEX = 19



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
        "traces_MMC",
        shape=(count, n_channels, trace_samples),
        dtype=DTYPE,
    )
    clean_dset = f.create_dataset(
        "traces_clean",
        shape=(count, n_channels, trace_samples),
        dtype=DTYPE,
    )
    white_dset = f.create_dataset(
        "traces_white",
        shape=(count, n_channels, trace_samples),
        dtype=DTYPE,
    )
    pink_dset = f.create_dataset(
        "traces_pink",
        shape=(count, n_channels, trace_samples),
        dtype=DTYPE,
    )
    return f, meta_dset, traces_dset, clean_dset, white_dset, pink_dset


def _scalar(v):
    return float(np.ravel(v)[0])


def _qp_select_and_sum(traces: np.ndarray, qp_mask: np.ndarray):
    qp_traces = traces[qp_mask]
    qp_sum = qp_traces.sum(axis=0, dtype=np.float64).astype(DTYPE, copy=False)
    return qp_traces, qp_sum


def _make_noise_generators(config):
    white_cfg = dict(config)
    white_cfg["noise_type"] = "white"
    pink_cfg = dict(config)
    pink_cfg["noise_type"] = "pink"
    return NoiseGenerator(white_cfg), NoiseGenerator(pink_cfg)


def _add_noise(noise_gen: NoiseGenerator, clean_traces: np.ndarray):
    noisy = np.empty_like(clean_traces, dtype=DTYPE)
    for ch in range(clean_traces.shape[0]):
        noise = noise_gen.generate_noise(clean_traces.shape[1]).astype(DTYPE, copy=False)
        noisy[ch] = clean_traces[ch] + noise
    return noisy


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
      - NR_traces_energy_<E>_pair_batch_<k>.h5 (noisy + clean + metadata, QP channels only)
      - NR_traces_energy_<E>_pair_qp_sum_batch_<k>.h5 (noisy + clean summed QP channel)
    Also writes additional traces with synthetic white and pink noise applied
    to the noiseless trace (same noise power config).
    """
    lts = LongTraceSimulator(CONFIG)
    white_gen, pink_gen = _make_noise_generators(CONFIG)

    batch_out = outdir / f"{RECOIL_TYPE}_traces_energy_{ENERGY}_pair_batch_{batch_idx:04d}.h5"
    sum_out = outdir / f"{RECOIL_TYPE}_traces_energy_{ENERGY}_pair_qp_sum_batch_{batch_idx:04d}.h5"

    traces, traces_clean, pos = _generate_pair(lts)
    event_traces = np.asarray(traces[0], dtype=DTYPE)
    event_traces_clean = np.asarray(traces_clean[0], dtype=DTYPE)
    n_channels, n_samples = event_traces.shape
    if n_samples != TRACE_SAMPLES:
        raise ValueError(f"Trace length {n_samples} != {TRACE_SAMPLES}")

    qp_mask = np.arange(n_channels) >= QP_CHANNEL_MIN_INDEX
    n_qp_channels = int(np.count_nonzero(qp_mask))
    if n_qp_channels == 0:
        raise ValueError("No QP channels found with index > 18")

    event_traces_qp, event_sum = _qp_select_and_sum(event_traces, qp_mask)
    event_traces_clean_qp, event_sum_clean = _qp_select_and_sum(event_traces_clean, qp_mask)
    event_traces_white_qp = _add_noise(white_gen, event_traces_clean_qp)
    event_traces_pink_qp = _add_noise(pink_gen, event_traces_clean_qp)
    event_sum_white = event_traces_white_qp.sum(axis=0, dtype=np.float64).astype(DTYPE, copy=False)
    event_sum_pink = event_traces_pink_qp.sum(axis=0, dtype=np.float64).astype(DTYPE, copy=False)

    h5f, meta_dset, trace_dset, clean_dset, white_dset, pink_dset = create_batch_h5(
        batch_out, batch_size, n_qp_channels, TRACE_SAMPLES
    )
    h5f_sum, meta_dset_sum, trace_dset_sum, clean_dset_sum, white_dset_sum, pink_dset_sum = create_batch_h5(
        sum_out, batch_size, 1, TRACE_SAMPLES
    )

    x0, y0, z0 = pos 
    meta_row = (
        _scalar(x0) if x0 is not None else np.nan,
        _scalar(y0) if y0 is not None else np.nan,
        _scalar(z0) if z0 is not None else np.nan,
        float(ENERGY),
        RECOIL_TYPE,
        bool(NO_NOISE),
        bool(QUANTIZE),
    )
    meta_dset[0] = meta_row
    meta_dset_sum[0] = meta_row
    trace_dset[0] = event_traces_qp
    clean_dset[0] = event_traces_clean_qp
    white_dset[0] = event_traces_white_qp
    pink_dset[0] = event_traces_pink_qp
    trace_dset_sum[0, 0] = event_sum
    clean_dset_sum[0, 0] = event_sum_clean
    white_dset_sum[0, 0] = event_sum_white
    pink_dset_sum[0, 0] = event_sum_pink
    del event_traces, traces, event_traces_clean

    for i in range(1, batch_size):
        ts_noisy, ts_clean, pos_i = _generate_pair(lts)
        ev_tr = np.asarray(ts_noisy[0], dtype=DTYPE)
        ev_clean = np.asarray(ts_clean[0], dtype=DTYPE)
        if ev_tr.shape[0] != n_channels:
            raise ValueError(f"Channel count {ev_tr.shape[0]} != {n_channels}")
        if ev_tr.shape[1] != TRACE_SAMPLES:
            raise ValueError(f"Trace length {ev_tr.shape[1]} != {TRACE_SAMPLES}")
        ev_tr_qp, ev_sum = _qp_select_and_sum(ev_tr, qp_mask)
        ev_clean_qp, ev_sum_clean = _qp_select_and_sum(ev_clean, qp_mask)
        ev_white_qp = _add_noise(white_gen, ev_clean_qp)
        ev_pink_qp = _add_noise(pink_gen, ev_clean_qp)
        ev_sum_white = ev_white_qp.sum(axis=0, dtype=np.float64).astype(DTYPE, copy=False)
        ev_sum_pink = ev_pink_qp.sum(axis=0, dtype=np.float64).astype(DTYPE, copy=False)
        x, y, z = pos_i 
        meta_row = (
            _scalar(x) if x is not None else np.nan,
            _scalar(y) if y is not None else np.nan,
            _scalar(z) if z is not None else np.nan,
            float(ENERGY),
            RECOIL_TYPE,
            bool(NO_NOISE),
            bool(QUANTIZE),
        )
        meta_dset[i] = meta_row
        meta_dset_sum[i] = meta_row
        trace_dset[i] = ev_tr_qp
        clean_dset[i] = ev_clean_qp
        white_dset[i] = ev_white_qp
        pink_dset[i] = ev_pink_qp
        trace_dset_sum[i, 0] = ev_sum
        clean_dset_sum[i, 0] = ev_sum_clean
        white_dset_sum[i, 0] = ev_sum_white
        pink_dset_sum[i, 0] = ev_sum_pink
        del ev_tr, ts_noisy, ev_clean

    h5f.flush()
    h5f.close()
    h5f_sum.flush()
    h5f_sum.close()
    return n_qp_channels


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
    for energy in ENERGIES:
        global ENERGY
        ENERGY = energy
        run_one_energy(OUTDIR)


if __name__ == "__main__":
    main()
