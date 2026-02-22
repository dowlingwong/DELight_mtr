import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import h5py

from TraceSimulator import LongTraceSimulator


# -------------------
# Paths & globals
# -------------------
CONFIG_PATH = "/home/dwong/DELight_mtr/trigger_study/wk29/training_config.yaml"
BASE_OUTDIR = Path("/ceph/dwong/work/threshold/pink_run2")  # base folder

NOISE_POWERS = [15.4, 42.8, 126.7, 472]

# -------------------
# Load simulator config
# -------------------
def read_yaml_to_dict(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# -------------------
# Config
# -------------------
TOTAL_SETS       = 50        # events per energy
TRACE_SAMPLES    = 16384
DTYPE            = np.float16

ENERGIES         = list(range(0, 101, 1))  # 0,2,...,58 (30 energies)

# Use 26 cores: each core handles one energy at a time
ENERGY_WORKERS   = 7

TYPE_RECOIL      = "NR"
PHONON_ONLY      = True
NO_NOISE         = False
QUANTIZE         = True
ROUTE_ALL_SIGNAL_TO_CH0 = False


def save_traces_h5(traces: np.ndarray, output_path: Path, *, energy: float, noise_power: float):
    """
    Store traces in an HDF5 file with minimal attributes.
    Dataset layout: (n_events, n_channels, TRACE_SAMPLES) with dtype `DTYPE`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traces = np.asarray(traces, dtype=DTYPE)

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "traces",
            data=traces,
            dtype=DTYPE,
            chunks=(1, traces.shape[1], traces.shape[2]),
        )
        f.attrs["energy"] = float(energy)
        f.attrs["noise_power"] = float(noise_power)
        f.attrs["n_channels"] = int(traces.shape[1])
        f.attrs["trace_samples"] = int(traces.shape[2])
        f.attrs["trace_dtype"] = np.dtype(DTYPE).name


# -------------------
# Worker: generate one energy (vectorized over TOTAL_SETS events)
# -------------------
def generate_one_energy(energy: float, config: dict, outdir: Path):
    """
    Generate exactly TOTAL_SETS events for this energy into traces_energy_<E>.h5.
    Uses vectorized generate: E is a length-TOTAL_SETS list.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    traces_out = outdir / f"traces_energy_{energy}.h5"

    # Build simulator in this process
    lts = LongTraceSimulator(config)

    # Vectorized energy: TOTAL_SETS events of same energy
    E_vec = [energy] * TOTAL_SETS

    # NOTE: keep extra args if your LongTraceSimulator.generate supports them
    ts, _ = lts.generate(
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

    save_traces_h5(
        ts,
        traces_out,
        energy=energy,
        noise_power=config.get("noise_power", np.nan),
    )

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
