import yaml
import numpy as np
import zstandard as zstd
import os
from TraceSimulator import TraceSimulator
from trace_IO import save_traces_to_zstd  # Make sure this function is available

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# --- Load YAML Config ---
def read_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

config = read_yaml_to_dict('../archive/config.yaml')
ts = TraceSimulator(config)

# --- Output Directories ---
BASE_OUTPUT_DIR = Path("/ceph/dwong/trigger_samples/large_samples")
NR_DIR = BASE_OUTPUT_DIR / "NR"
ER_DIR = BASE_OUTPUT_DIR / "ER"
NR_DIR.mkdir(parents=True, exist_ok=True)
ER_DIR.mkdir(parents=True, exist_ok=True)

# --- Generate and Save Function ---
def generate_and_save_traces(energy):
    n_sets = 300
    for trace_type in ['NR', 'ER']:
        all_traces = []

        print(f"[INFO] Generating {n_sets} traces for energy={energy}, type={trace_type}")

        for _ in range(n_sets):
            try:
                trace = ts.generate(
                    E=energy,
                    x=-40, y=80, z=-1800,
                    no_noise=False,
                    type_recoil=trace_type,
                    quantize=True,
                    phonon_only=False
                )
                all_traces.append(np.asarray(trace[0], dtype=np.float16))
            except Exception as e:
                print(f"[ERROR] Failed to generate trace for E={energy}, type={trace_type}: {e}")

        # Determine output path
        output_dir = NR_DIR if trace_type == 'NR' else ER_DIR
        output_path = output_dir / f"traces_energy_{energy}.zst"

        try:
            save_traces_to_zstd(all_traces, output_path)
            print(f"[DONE] Saved {trace_type} traces for energy={energy} to {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save traces for E={energy}, type={trace_type}: {e}")

# --- Main Execution ---
def main():
    energy_values = list(range(2, 151, 2))  # 2 to 150 keV in steps of 2
    max_threads = 15

    print("[START] Beginning trace generation...")
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(generate_and_save_traces, energy): energy
            for energy in energy_values
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Traces"):
            energy = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Trace generation failed for energy {energy}: {e}")

    print("[COMPLETE] All trace sets generated and saved.")


if __name__ == "__main__":
    main()
