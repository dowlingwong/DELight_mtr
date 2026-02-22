import os
import re
from pathlib import Path

# Limit BLAS/OpenMP thread usage before importing numpy/scipy stack.
THREAD_LIMIT = "50"
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = THREAD_LIMIT

import numpy as np
import h5py

# Import optimized EMPCA
import sys
# wk3/train -> wk3 -> PCA_dev -> reusable
reusable_dir = Path(__file__).resolve().parent.parent.parent / "reusable"
sys.path.insert(0, str(reusable_dir))
from empca_TCY_optimized import EMPCA, ti_rfft


# ----------------------------
# 1) Baseline correction
# ----------------------------

def baseline_correct_per_trace(X_time, pretrigger=4000, method="mean"):
    X_time = np.asarray(X_time, dtype=np.float64)
    if X_time.ndim != 2:
        raise ValueError(f"X_time must be 2D; got {X_time.shape}")
    if not (1 <= pretrigger <= X_time.shape[1]):
        raise ValueError("pretrigger must be within [1, n_time]")

    pre = X_time[:, :pretrigger]
    if method == "mean":
        baseline = np.mean(pre, axis=1)
    elif method == "median":
        baseline = np.median(pre, axis=1)
    else:
        raise ValueError("method must be 'mean' or 'median'")

    X0 = X_time - baseline[:, None]
    return X0, baseline


# ----------------------------
# 2) Frequency-domain shift-invariant transform
# ----------------------------

def to_shift_invariant_spectrum(X_time):
    X_tilde = ti_rfft(X_time)
    if X_tilde.ndim != 2:
        raise RuntimeError("Unexpected output shape from ti_rfft")
    return X_tilde


# ----------------------------
# 3) Build weights vector from PSD (optimized EMPCA)
# ----------------------------

def make_inverse_psd_weights(noise_psd, eps=1e-18):
    noise_psd = np.asarray(noise_psd, dtype=np.float64)
    if noise_psd.ndim != 1:
        raise ValueError(f"noise_psd must be 1D; got {noise_psd.shape}")
    inv = 1.0 / (noise_psd + eps)
    return inv


# ----------------------------
# 4) Train EM-PCA
# ----------------------------

def train_empca_paper_style(
    X_time,
    noise_psd,
    n_comp=8,
    pretrigger=4000,
    baseline_method="mean",
    n_iter=50,
    mode="fast",
    window=15,
    polyord=3,
    deriv=0,
):
    X0, baseline = baseline_correct_per_trace(
        X_time, pretrigger=pretrigger, method=baseline_method
    )

    X_tilde = to_shift_invariant_spectrum(X0)

    n_freq = X_tilde.shape[1]
    if len(noise_psd) != n_freq:
        raise ValueError(
            f"PSD length mismatch: PSD has {len(noise_psd)} bins but data has {n_freq}. "
            f"For n_time={X_time.shape[1]}, expected n_freq = n_time//2 + 1."
        )

    w = make_inverse_psd_weights(noise_psd)

    pca = EMPCA(n_comp=n_comp)
    chi2s = pca.fit(
        X_tilde,
        w,
        n_iter=n_iter,
        mode=mode,
        window=window,
        polyord=polyord,
        deriv=deriv,
        verbose=False,
    )

    return pca, chi2s, baseline, X_tilde, w


# ----------------------------
# Utilities
# ----------------------------

def clip_psd_for_weights(psd, psd_floor=None, w_max=None):
    psd_eff = np.asarray(psd, dtype=float).copy()
    if psd_floor is not None:
        psd_eff = np.maximum(psd_eff, float(psd_floor))
    if w_max is not None:
        psd_eff = np.maximum(psd_eff, 1.0 / float(w_max))
    return psd_eff


def baseline_rms_filter(X, pretrigger, frac=0.02):
    pre = X[:, :pretrigger]
    pre0 = pre - np.mean(pre, axis=1, keepdims=True)
    rms = np.std(pre0, axis=1)
    if frac <= 0:
        keep = np.ones(len(X), dtype=bool)
        return X, keep, rms
    thr = np.quantile(rms, 1.0 - frac)
    keep = rms <= thr
    return X[keep], keep, rms


def template_snr_spectrum(template, noise_psd, sampling_frequency, eps=1e-18):
    """
    Build a one-sided SNR-like spectrum using rFFT(template) / noise_psd.
    This follows the OptimumFilter convention where the matched filter uses
    template_fft.conj() / noise_psd_unfolded; here we use one-sided bins to
    match ti_rfft output lengths.
    """
    template = np.asarray(template, dtype=np.float64)
    if template.ndim != 1:
        raise ValueError("template must be 1D")

    template_fft = np.fft.rfft(template) / float(sampling_frequency)
    if len(noise_psd) != len(template_fft):
        raise ValueError(
            f"PSD length mismatch: PSD has {len(noise_psd)} bins but template rfft has {len(template_fft)} bins."
        )

    snr = (np.abs(template_fft) ** 2) / (noise_psd + eps)
    return snr


def train_with_early_stopping(X_train, weights, n_comp, *, n_iter_max, rtol, patience, **kwargs):
    warm_start_supported = kwargs.pop("warm_start_supported", False)

    chi2s_all = []

    if not warm_start_supported:
        pca, chi2s, baselines, X_tilde, w = train_empca_paper_style(
            X_train,
            weights,
            n_comp=n_comp,
            n_iter=n_iter_max,
            **kwargs,
        )
        chi2s_all = list(chi2s)
        return pca, np.array(chi2s_all), {"baselines": baselines, "X_tilde": X_tilde, "w": w}

    pca_init = None
    no_improve = 0
    last = None
    chunk = 5

    for it0 in range(0, n_iter_max, chunk):
        pca, chi2s, baselines, X_tilde, w = train_empca_paper_style(
            X_train,
            weights,
            n_comp=n_comp,
            n_iter=min(chunk, n_iter_max - it0),
            pca_init=pca_init,
            **kwargs,
        )
        pca_init = pca
        for c in chi2s:
            chi2s_all.append(float(c))
            if last is not None:
                rel = (last - c) / max(abs(last), 1e-12)
                if rel < rtol:
                    no_improve += 1
                else:
                    no_improve = 0
            last = c
            if no_improve >= patience:
                return pca_init, np.array(chi2s_all), {"baselines": baselines, "X_tilde": X_tilde, "w": w}

    return pca_init, np.array(chi2s_all), {"baselines": baselines, "X_tilde": X_tilde, "w": w}


# ----------------------------
# Config
# ----------------------------

OUTDIR = Path("/ceph/dwong/trigger_samples/PCA_QP/main")
ENERGY_EV = 500
N_BATCHES = 10

REUSABLE_DIR = Path(__file__).resolve().parent.parent.parent / "reusable"
WEIGHT_DIR = REUSABLE_DIR / "weight"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define the 3 noise scenarios
NOISE_CONFIGS = {
    "mmc": {
        "psd": Path("/ceph/dwong/delight/noise_psd_xray.npy"),
        "snr2": WEIGHT_DIR / "qp_snr2_weight_MMC.npy",
        "trace_key": "traces_MMC",
    },
    "white": {
        "psd": WEIGHT_DIR / "noise_psd_white.npy",
        "snr2": WEIGHT_DIR / "qp_snr2_weight_white.npy",
        "trace_key": "traces_white",
    },
    "pink": {
        "psd": WEIGHT_DIR / "noise_psd_pink.npy",
        "snr2": WEIGHT_DIR / "qo_snr2_weight_pink.npy",
        "trace_key": "traces_pink",
    },
}

CFG = {
    "n_comp_max": 8,
    "n_iter_max": 100,
    "early_stop_rtol": 1e-5,
    "early_stop_patience": 4,
    "n_restarts": 5,
    "pretrigger": 4000,
    "baseline_method": "mean",
    "mode": "fast",
    "window": 15,
    "polyord": 3,
    "deriv": 0,
    "remove_baseline_rms_frac": 0.02,
    "psd_floor": None,
    "w_max": 1e6,
    "val_frac": 0.25,
    "seed": 0,
}


# ----------------------------
# Load sum-channel traces
# ----------------------------

def batch_id(path: Path) -> int:
    match = re.search(r"_batch_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse batch id from {path.name}")
    return int(match.group(1))


def load_sum_traces(outdir: Path, energy_ev: int, n_batches: int, trace_key: str = "traces"):
    sum_pattern = f"NR_traces_energy_{energy_ev}_pair_qp_sum_batch_*.h5"
    sum_files = sorted(outdir.glob(sum_pattern))
    if not sum_files:
        raise FileNotFoundError(f"No sum batch files found in {outdir} for energy {energy_ev} eV")

    sum_files = sorted(sum_files, key=batch_id)
    if len(sum_files) < n_batches:
        raise FileNotFoundError(f"Found {len(sum_files)} batches; expected at least {n_batches}.")

    picked = sum_files[:n_batches]
    sum_list = []
    for p in picked:
        print(f"loading batch {batch_id(p):04d}: {p.name} [{trace_key}]")
        with h5py.File(p, "r") as f:
            if trace_key not in f:
                raise KeyError(f"Key {trace_key} not found in {p}")
            sum_list.append(f[trace_key][:].astype(np.float64))

    sum_traces = np.concatenate(sum_list, axis=0)
    X_sum = sum_traces[:, 0, :]
    return X_sum, [str(p) for p in picked]


def main():
    # Loop over noise configurations
    for noise_name, noise_cfg in NOISE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"DTO Training for Noise Type: {noise_name}")
        print(f"{'='*60}")

        psd_path = noise_cfg["psd"]
        snr2_path = noise_cfg["snr2"]
        trace_key = noise_cfg["trace_key"]

        # 1) Load Traces specific to this noise type
        print(f"Loading traces for {noise_name} using key '{trace_key}'")
        try:
            X_sum, sum_paths = load_sum_traces(OUTDIR, ENERGY_EV, N_BATCHES, trace_key=trace_key)
        except Exception as e:
            print(f"Failed to load traces for {noise_name}: {e}")
            continue
            
        print("X_sum shape:", X_sum.shape)

        # Filter out noisy events
        X0, keep_mask, _ = baseline_rms_filter(
            X_sum, pretrigger=CFG["pretrigger"], frac=CFG["remove_baseline_rms_frac"]
        )
        print(f"Kept {len(X0)} / {len(X_sum)} events after baseline-RMS filtering.")

        # Train/val split
        rng = np.random.default_rng(CFG["seed"])
        idx = np.arange(len(X0))
        rng.shuffle(idx)
        n_val = int(CFG["val_frac"] * len(idx))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train = X0[train_idx]
        # X_val = X0[val_idx] 

        # --- Load PSD ---
        if not psd_path.exists():
            print(f"WARNING: PSD file not found: {psd_path}. Skipping {noise_name}.")
            continue
        
        psd_arr = np.load(psd_path)
        # Handle cases where PSD might be (2, N) [freq, psd] or just (N,) [psd]
        psd = psd_arr[1] if psd_arr.ndim == 2 and psd_arr.shape[0] == 2 else psd_arr
        psd = psd.astype(np.float64)
       
        # Clip PSD for weighting
        psd_eff = clip_psd_for_weights(psd, psd_floor=CFG["psd_floor"], w_max=CFG["w_max"])

        # ----------------------------
        # Model A: 1/PSD weights
        # ----------------------------
        results = []
        for r in range(CFG["n_restarts"]):
            seed = CFG["seed"] + 100 * r
            np.random.seed(seed)
            print()
            print(f"[{noise_name}/1overPSD] Restart {r+1}/{CFG['n_restarts']} (seed={seed})")
            
            weights_inv_psd = make_inverse_psd_weights(psd_eff)
            
            try:
                pca, chi2s, extras = train_with_early_stopping(
                    X_train,
                    weights_inv_psd,
                    n_comp=CFG["n_comp_max"],
                    n_iter_max=CFG["n_iter_max"],
                    rtol=CFG["early_stop_rtol"],
                    patience=CFG["early_stop_patience"],
                    baseline_method=CFG["baseline_method"],
                    pretrigger=CFG["pretrigger"],
                    mode=CFG["mode"],
                    window=CFG["window"],
                    polyord=CFG["polyord"],
                    deriv=CFG["deriv"],
                )
                print(f"  Final chi2: {chi2s[-1]:.6g} (iters={len(chi2s)})")
                results.append({"pca": pca, "chi2s": chi2s, "extras": extras, "seed": seed})
            except ValueError as e:
                print(f"  Training failed for {noise_name} 1/PSD: {e}")
                # Potentially try resampling PSD if size mismatch, but for now just skip/fail safely
                break

        if results:
            best = min(results, key=lambda d: d["chi2s"][-1])
            print()
            print(f"[{noise_name}/1overPSD] Selected best model by final chi2.")

            # Save 1/PSD model
            import pickle
            model_out_path = MODELS_DIR / f"PSD_run1_sum_{noise_name}.pkl"
            model_artifact = {
                "pca": best["pca"],
                "cfg": CFG,
                "psd_path": str(psd_path),
                "psd_eff": psd_eff,
                "weight_type": "1/PSD",
                "keep_mask": keep_mask,
                "dataset_paths": sum_paths,
                "dataset": "sum",
                "energy_ev": ENERGY_EV,
                "n_batches": N_BATCHES,
                "noise_type": noise_name,
                "trace_key": trace_key,
            }
            with open(model_out_path, "wb") as f:
                pickle.dump(model_artifact, f)
            print("saved model to", model_out_path)

        # ----------------------------
        # Model B: Template SNR^2 weights
        # ----------------------------
        if not snr2_path.exists():
            print(f"WARNING: SNR2 weights not found at {snr2_path}. Skipping Model B for {noise_name}.")
        else:
            snr2_weights = np.load(snr2_path).astype(np.float64)
            # Check shape compatibility
            n_freq_data = X_sum.shape[1] // 2 + 1
            if snr2_weights.shape[0] != n_freq_data:
                 print(f"WARNING: SNR2 weight length {snr2_weights.shape[0]} != data freq bins {n_freq_data}.")
            
            results_b = []
            for r in range(CFG["n_restarts"]):
                seed = CFG["seed"] + 100 * r
                np.random.seed(seed)
                print()
                print(f"[{noise_name}/SNR2] Restart {r+1}/{CFG['n_restarts']} (seed={seed})")
                
                try:
                    pca, chi2s, extras = train_with_early_stopping(
                        X_train,
                        snr2_weights,
                        n_comp=CFG["n_comp_max"],
                        n_iter_max=CFG["n_iter_max"],
                        rtol=CFG["early_stop_rtol"],
                        patience=CFG["early_stop_patience"],
                        baseline_method=CFG["baseline_method"],
                        pretrigger=CFG["pretrigger"],
                        mode=CFG["mode"],
                        window=CFG["window"],
                        polyord=CFG["polyord"],
                        deriv=CFG["deriv"],
                    )
                    print(f"  Final chi2: {chi2s[-1]:.6g} (iters={len(chi2s)})")
                    results_b.append({"pca": pca, "chi2s": chi2s, "extras": extras, "seed": seed})
                except Exception as e:
                    print(f"  Training failed for {noise_name} SNR2: {e}")
                    break

            if results_b:
                best = min(results_b, key=lambda d: d["chi2s"][-1])
                print()
                print(f"[{noise_name}/SNR2] Selected best model by final chi2.")

                # Save SNR2-weighted model
                model_out_path = MODELS_DIR / f"SNR2_run1_sum_{noise_name}.pkl"
                model_artifact = {
                    "pca": best["pca"],
                    "cfg": CFG,
                    "snr2_weight_path": str(snr2_path),
                    "weight_type": "SNR2=|H|^2/PSD",
                    "keep_mask": keep_mask,
                    "dataset_paths": sum_paths,
                    "dataset": "sum",
                    "energy_ev": ENERGY_EV,
                    "n_batches": N_BATCHES,
                    "noise_type": noise_name,
                    "trace_key": trace_key,
                }
                with open(model_out_path, "wb") as f:
                    pickle.dump(model_artifact, f)
                print("saved model to", model_out_path)

if __name__ == "__main__":
    main()
