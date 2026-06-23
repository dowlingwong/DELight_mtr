"""
Energy sweep runner with dual OptimumFilters:
- First 19 channels use submerged-channel template
- Last 37 channels use vacuum-channel template

Inputs/outputs (wk29):
    in_pattern   = /ceph/dwong/work/threshold/noise_{P}/traces_energy_{E}.h5
    sub_template = /home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/sub_ch_template.npy
    vac_template = /home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy
    noise_psd    = trigger_study/wk29/pink_psd.npy (one-sided, length 16385)
    fs           = 250000
    repeats      = 50 traces per energy
    channels     = 56, template_len = 32768
    energies     = range(0, 101, 1)
    out_dir      = /ceph/dwong/work/threshold/noise_{P}/fit_with_shift_amp
"""

from __future__ import annotations
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import h5py

THIS_DIR = Path(__file__).resolve().parent
UTILS_DIR = THIS_DIR.parent / "utils"
sys.path.insert(0, str(UTILS_DIR))

from archive.trigger_study.archive.wk29.OptimumFilter import OptimumFilter

# -------------------
# Worker process globals
# -------------------
_WORKER_OF_SUB = None
_WORKER_OF_VAC = None
_REANCHOR_EVERY = None


def _worker_init(sub_template: np.ndarray, vac_template: np.ndarray,
                 noise_psd: np.ndarray, fs: float, reanchor_every: int | None):
    """Initializer runs once per worker. Builds two OptimumFilters."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    global _WORKER_OF_SUB, _WORKER_OF_VAC, _REANCHOR_EVERY
    _WORKER_OF_VAC = OptimumFilter(vac_template, noise_psd, fs)
    _WORKER_OF_SUB = OptimumFilter(sub_template, noise_psd, fs)
    _REANCHOR_EVERY = reanchor_every


def _max_amp_for_trace(args) -> float:
    """
    Pick correct OptimumFilter depending on channel index.
    Uses fit_with_shift, returns a single amplitude per trace.
    """
    trace_1d, ch = args
    x = np.ascontiguousarray(trace_1d, dtype=np.float64)
    if ch < 19:  # first 19 channels: submerged OF
        amp, _, _ = _WORKER_OF_SUB.fit_with_shift(x)
    else:        # remaining 37 channels: vacuum OF
        amp, _, _ = _WORKER_OF_VAC.fit_with_shift(x)
    return float(amp)


# -------------------
# Core computation
# -------------------
def compute_max_amplitudes(traces: np.ndarray, sub_template: np.ndarray, vac_template: np.ndarray,
                           noise_psd: np.ndarray, fs: float, *, max_workers: int = 56,
                           reanchor_every: int | None = None, show_progress: bool = True) -> np.ndarray:
    """Compute OF amplitude per trace with concurrency (fit_with_shift). Returns (R, T)."""
    R, T, L = traces.shape
    n_workers = min(int(max_workers), os.cpu_count() or 1)
    amps_out = np.empty((R, T), dtype=np.float64)

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
            amps_out[r, :] = np.asarray(row, dtype=np.float64)

    return amps_out


# -------------------
# Loader for traces from HDF5
# -------------------
def load_traces_from_h5(input_path: Path, repeats: int, expected_channels: int, target_len: int) -> np.ndarray:
    """
    Load traces from an HDF5 file created by make_traces.py.
    Returns shape (repeats, expected_channels, target_len) as float64.
    If trace length differs from target_len, it is padded (zeros) or truncated.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with h5py.File(input_path, "r") as f:
        traces = f["traces"][:]

    if traces.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {traces.shape[1]} in {input_path}")
    if traces.shape[0] < repeats:
        raise ValueError(f"Requested {repeats} repeats but only {traces.shape[0]} available in {input_path}")

    traces = np.asarray(traces[:repeats], dtype=np.float64)
    current_len = traces.shape[2]
    if current_len > target_len:
        traces = traces[:, :, :target_len]
    elif current_len < target_len:
        pad_width = target_len - current_len
        traces = np.pad(traces, ((0, 0), (0, 0), (0, pad_width)), mode="constant")

    return traces


def save_amplitudes_h5(output_path: Path, amps: np.ndarray, *, energy: float, noise_power: float):
    """
    Save amplitudes array shape (repeats, n_channels) to HDF5 with minimal attrs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("amplitudes", data=amps, dtype=np.float64, chunks=True)
        f.attrs["energy"] = float(energy)
        f.attrs["noise_power"] = float(noise_power)
        f.attrs["n_repeats"] = int(amps.shape[0])
        f.attrs["n_channels"] = int(amps.shape[1])
        f.attrs["method"] = "fit_with_shift"


# -------------------
# Main runner
# -------------------
def main() -> int:
    NOISE_POWERS = [15.4, 42.8, 126.7, 472]

    sub_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/sub_ch_template.npy")
    vac_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy")

    psd_path = THIS_DIR / "pink_psd.npy"
    noise_psd = np.load(psd_path)

    fs = 250000.0
    repeats = 50
    n_traces = 56
    energies = list(range(0, 101, 1))

    template_len = int(sub_template.shape[0])
    if vac_template.shape[0] != template_len:
        raise ValueError("Sub and vac templates must have the same length.")
    if noise_psd.shape[0] != template_len // 2 + 1:
        raise ValueError(f"PSD length {noise_psd.shape[0]} does not match template length {template_len}.")

    # Loop over all noise power directories
    for noise_power in NOISE_POWERS:
        noise_str = f"{noise_power:.1f}"  # e.g., "13.2"

        # Pattern for input traces
        in_pattern = f"/ceph/dwong/work/threshold/pink_run1/noise_{noise_str}/traces_energy_{{energy}}.h5"

        # Output directory for this noise power
        out_dir = Path(f"/ceph/dwong/work/threshold/pink_run1/noise_{noise_str}/fit_with_shift_amp")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing noise power {noise_str} ===")

        for E in tqdm(energies, desc=f"Noise {noise_str} | Energies", unit="E"):
            in_path = Path(in_pattern.format(energy=E))
            out_path = out_dir / f"amplitudes_energy_{E}.h5"

            try:
                traces = load_traces_from_h5(
                    in_path,
                    repeats=repeats,
                    expected_channels=n_traces,
                    target_len=template_len,
                )
            except FileNotFoundError:
                print(f"[skip] Input file not found: {in_path}")
                continue

            amps = compute_max_amplitudes(
                traces=traces,
                sub_template=sub_template,
                vac_template=vac_template,
                noise_psd=noise_psd,
                fs=fs,
                max_workers=60,
                reanchor_every=None,
                show_progress=True,
            )

            save_amplitudes_h5(out_path, amps, energy=E, noise_power=noise_power)
            print(f"Saved {out_path} with shape {amps.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
