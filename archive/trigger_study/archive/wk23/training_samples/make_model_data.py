import yaml
import numpy as np
from TraceSimulator import LongTraceSimulator

import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# -------------------
# Load config and simulator
# -------------------
def read_yaml_to_dict(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

config = read_yaml_to_dict("/ceph/dwong/delight_conf/ETP_config_v2.yaml")
lts = LongTraceSimulator(config)

# -------------------
# Config
# -------------------
N_SETS = 60                   # traces per position
TRACE_SAMPLES = 300_000         # must match your long-trace config
DTYPE = np.float16
COMPRESSION_LEVEL = 15
MAX_THREADS = 40
ENERGY_FIXED = 50               # eV
Z_FIXED = -1700.0               # z fixed as requested
OUTDIR = Path("/ceph/dwong/trigger_samples/v2_300k_xy_50ev")

# -------------------
# MMC positions: use ONLY x,y; filenames use indices 20..36
# (taken from your table rows 2020..2036)
# -------------------
positions_xy = {
    0: (-109.97308,  -61.80239),
    1: ( -75.98205,  -87.73651),
    2: ( -37.99103, -109.67064),
    3: (   0.00000, -125.60477),
    4: (-113.97308,  -21.93413),
    5: ( -75.98205,  -43.86826),
    6: ( -37.99103,  -65.80239),
    7: (   0.00000,  -87.73651),
    8: (  37.99103, -109.67064),
    9: (-113.97308,   21.93413),
    10: ( -75.98205,    0.00000),
    11: ( -37.99103,  -21.93413),
    12: (   0.00000,  -43.86826),
    13: (  37.99103,  -65.80239),
    14: (  75.98205,  -87.73651),
    15: (-109.97308,   61.80239),
    16: ( -75.98205,   43.86826),
    17: ( -37.99103,   21.93413),
    18: (   0.00000,    0.00000),
    19: (  37.99103,  -21.93413),
    20: (  75.98205,  -43.86826),
    21: ( 109.97308,  -61.80239),
    22: ( -75.98205,   87.73651),
    23: ( -37.99103,   65.80239),
    24: (   0.00000,   43.86826),
    25: (  37.99103,   21.93413),
    26: (  75.98205,    0.00000),
    27: ( 113.97308,  -21.93413),
    28: ( -37.99103,  109.67064),
    29: (   0.00000,   87.73651),
    30: (  37.99103,   65.80239),
    31: (  75.98205,   43.86826),
    32: ( 113.97308,   21.93413),
    33: (   0.00000,  125.60477),
    34: (  37.99103,  109.67064),
    35: (  75.98205,   87.73651),
    36: ( 109.97308,   61.80239),
}   

# -------------------
# Helpers
# -------------------
def _shuffle_bytes(arr: np.ndarray) -> bytes:
    return arr.view(np.uint8).reshape(-1, arr.itemsize).T.tobytes()

def save_traces_to_zstd(traces, output_path: Path,
                        dtype=DTYPE, trace_shape=None, compression_level=COMPRESSION_LEVEL):
    if trace_shape is None:
        trace_shape = traces[0].shape
    all_data = bytearray()
    for t in traces:
        shuffled = _shuffle_bytes(np.asarray(t, dtype=dtype))
        all_data.extend(shuffled)
    comp = zstd.ZstdCompressor(level=compression_level)
    data = comp.compress(bytes(all_data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)

def save_masks_to_zstd(masks: np.ndarray, output_path: Path,
                       mask_len: int, compression_level=COMPRESSION_LEVEL):
    if masks.dtype != np.bool_:
        masks = masks.astype(np.bool_)
    bytes_per_row = (mask_len + 7) // 8
    rows = []
    for row in masks:
        packed = np.packbits(row, bitorder="little")
        if packed.size < bytes_per_row:
            pad = np.zeros(bytes_per_row - packed.size, dtype=np.uint8)
            packed = np.concatenate([packed, pad])
        rows.append(packed.tobytes())
    blob = b"".join(rows)
    comp = zstd.ZstdCompressor(level=compression_level)
    data = comp.compress(blob)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)

# -------------------
# Generation worker per MMC index
# -------------------
def generate_and_save_traces(mmc_idx: int, x: float, y: float):
    """
    Generates N_SETS traces at (x,y,Z_FIXED) with fixed energy,
    saves to traces_<mmc_idx>.zst and masks_<mmc_idx>.zst.
    """
    all_traces, all_masks = [], []

    for _ in range(N_SETS):
        traces, signal_mask = lts.generate(
            E=ENERGY_FIXED,
            x=x,
            y=y,
            z=Z_FIXED,  # force z = -1700
            type_recoil="NR",
            phonon_only=True,
            no_noise=False,
            quantize=True,
            return_signal_mask=True,
        )
        event_traces = np.asarray(traces[0], dtype=DTYPE)   # (n_channels, TRACE_SAMPLES)
        event_mask   = np.asarray(signal_mask[0], dtype=bool)  # (n_channels,)

        all_traces.append(event_traces)
        all_masks.append(event_mask)

    n_channels = all_traces[0].shape[0]
    trace_shape = (n_channels, TRACE_SAMPLES)

    traces_out_path = OUTDIR / f"traces_{mmc_idx}.zst"
    masks_out_path  = OUTDIR / f"masks_{mmc_idx}.zst"

    save_traces_to_zstd(all_traces, traces_out_path, dtype=DTYPE,
                        trace_shape=trace_shape, compression_level=COMPRESSION_LEVEL)
    save_masks_to_zstd(np.vstack(all_masks), masks_out_path,
                       mask_len=n_channels, compression_level=COMPRESSION_LEVEL)

# -------------------
# Driver
# -------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(generate_and_save_traces, idx, xy[0], xy[1]): idx
            for idx, xy in positions_xy.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="MMC generation"):
            idx = futures[fut]
            try:
                fut.result()
            except Exception as ex:
                print(f"[MMC {idx}] FAILED: {ex}")
                raise

if __name__ == "__main__":
    main()
