#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_SIM_ROOT = Path("/home/dwong/software/TraceSimulator")
if str(TRACE_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_SIM_ROOT))

from TraceSimulator.NoiseGenerator import NoiseGenerator


DEFAULT_CFG = REPO_ROOT / "reusable" / "PCA_config.yaml"
DEFAULT_PAIR = Path(
    "/ceph/dwong/trigger_samples/PCA_QP/main/NR_traces_energy_500_pair_qp_sum_batch_0000.h5"
)
DEFAULT_OUTDIR = REPO_ROOT / "wk5" / "figures" / "noise_slide_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export slide-ready plots and data for PSD shape, noise-only traces, "
            "and clean/MMC/pink/white trace overlays."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--pair-file", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--event-idx", type=int, default=0)
    parser.add_argument("--channel-idx", type=int, default=0)
    parser.add_argument("--psd-draws", type=int, default=200)
    parser.add_argument("--plot-start", type=int, default=0)
    parser.add_argument("--plot-samples", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def periodogram_one_sided(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    n = x.shape[-1]
    xf = np.fft.rfft(x, axis=-1)
    psd = (np.abs(xf) ** 2) / (fs * n)
    if n % 2 == 0:
        psd[..., 1:-1] *= 2.0
    else:
        psd[..., 1:] *= 2.0
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, psd


def generate_noise_batch(config: dict, noise_type: str, n_samples: int, n_draws: int) -> np.ndarray:
    c = dict(config)
    c["noise_type"] = noise_type
    ng = NoiseGenerator(c)
    return np.array([ng.generate_noise(n_samples) for _ in range(n_draws)], dtype=np.float64)


def load_pair_trace(path: Path, event_idx: int, channel_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Pair file does not exist: {path}")
    with h5py.File(path, "r") as h5f:
        if "traces_clean" not in h5f or "traces_MMC" not in h5f:
            raise KeyError("Expected datasets 'traces_clean' and 'traces_MMC' not found.")
        clean_event = np.asarray(h5f["traces_clean"][event_idx], dtype=np.float32)
        mmc_event = np.asarray(h5f["traces_MMC"][event_idx], dtype=np.float32)

    if clean_event.ndim == 1:
        clean = clean_event
        mmc = mmc_event
    elif clean_event.ndim == 2:
        if channel_idx < 0 or channel_idx >= clean_event.shape[0]:
            raise IndexError(
                f"channel_idx={channel_idx} out of range for event with {clean_event.shape[0]} channels"
            )
        clean = clean_event[channel_idx]
        mmc = mmc_event[channel_idx]
    else:
        raise ValueError(f"Unexpected trace dimensionality: {clean_event.shape}")

    return clean.astype(np.float32, copy=False), mmc.astype(np.float32, copy=False)


def save_data_tables(
    outdir: Path,
    sample_idx: np.ndarray,
    freq: np.ndarray,
    psd_curves: dict[str, np.ndarray],
    clean: np.ndarray,
    mmc: np.ndarray,
    white_trace: np.ndarray,
    pink_trace: np.ndarray,
    mmc_noise: np.ndarray,
    white_noise: np.ndarray,
    pink_noise: np.ndarray,
) -> None:
    psd_df = pd.DataFrame(
        {
            "frequency_hz": freq,
            "psd_mmc_custom": psd_curves["mmc_custom"],
            "psd_white": psd_curves["white"],
            "psd_pink": psd_curves["pink"],
        }
    )
    psd_df.to_csv(outdir / "psd_shape_data.csv", index=False)

    traces_df = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "clean": clean,
            "mmc": mmc,
            "white": white_trace,
            "pink": pink_trace,
        }
    )
    traces_df.to_csv(outdir / "trace_overlay_data.csv", index=False)

    noise_df = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "noise_mmc": mmc_noise,
            "noise_white": white_noise,
            "noise_pink": pink_noise,
        }
    )
    noise_df.to_csv(outdir / "noise_only_data.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"series": "noise_mmc", "variance": float(np.var(mmc_noise)), "rms": float(np.sqrt(np.mean(mmc_noise**2)))},
            {"series": "noise_white", "variance": float(np.var(white_noise)), "rms": float(np.sqrt(np.mean(white_noise**2)))},
            {"series": "noise_pink", "variance": float(np.var(pink_noise)), "rms": float(np.sqrt(np.mean(pink_noise**2)))},
        ]
    )
    summary_df.to_csv(outdir / "noise_summary_stats.csv", index=False)

    np.savez_compressed(
        outdir / "noise_slide_assets.npz",
        sample_idx=sample_idx,
        frequency_hz=freq,
        psd_mmc_custom=psd_curves["mmc_custom"],
        psd_white=psd_curves["white"],
        psd_pink=psd_curves["pink"],
        clean=clean,
        mmc=mmc,
        white=white_trace,
        pink=pink_trace,
        noise_mmc=mmc_noise,
        noise_white=white_noise,
        noise_pink=pink_noise,
    )


def plot_psd_shape(outdir: Path, freq: np.ndarray, psd_curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(freq[1:], psd_curves["mmc_custom"][1:], label="MMC (custom PSD)", linewidth=2.0, color="#1f77b4")
    ax.loglog(freq[1:], psd_curves["white"][1:], label="White", linewidth=2.0, color="#ff7f0e")
    ax.loglog(freq[1:], psd_curves["pink"][1:], label="Pink", linewidth=2.0, color="#d62728")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("One-sided PSD")
    ax.set_title("PSD Shape Only")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "slide_psd_shape.png", dpi=220)
    plt.close(fig)


def plot_noise_only(
    outdir: Path,
    x_axis: np.ndarray,
    mmc_noise: np.ndarray,
    white_noise: np.ndarray,
    pink_noise: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    ax.plot(x_axis, mmc_noise, label="MMC noise", alpha=0.5, linewidth=1.1, color="#1f77b4")
    ax.plot(x_axis, white_noise, label="White noise", alpha=0.5, linewidth=1.1, color="#ff7f0e")
    ax.plot(x_axis, pink_noise, label="Pink noise", alpha=0.5, linewidth=1.1, color="#d62728")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Noise Traces (Half Transparent)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "slide_noise_only.png", dpi=220)
    plt.close(fig)


def plot_trace_overlay(
    outdir: Path,
    x_axis: np.ndarray,
    clean: np.ndarray,
    mmc: np.ndarray,
    white_trace: np.ndarray,
    pink_trace: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    ax.plot(x_axis, clean, label="Clean", linewidth=2.0, color="black")
    ax.plot(x_axis, mmc, label="MMC", alpha=0.5, linewidth=1.2, color="#1f77b4")
    ax.plot(x_axis, pink_trace, label="Pink", alpha=0.5, linewidth=1.2, color="#d62728")
    ax.plot(x_axis, white_trace, label="White", alpha=0.5, linewidth=1.2, color="#ff7f0e")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Clean + MMC + Pink + White (Half Transparent Noisy Traces)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "slide_trace_overlay.png", dpi=220)
    plt.close(fig)


def plot_slide_1_combo(
    outdir: Path,
    freq: np.ndarray,
    psd_curves: dict[str, np.ndarray],
    x_axis: np.ndarray,
    mmc_noise: np.ndarray,
    white_noise: np.ndarray,
    pink_noise: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.2))

    axes[0].loglog(freq[1:], psd_curves["mmc_custom"][1:], label="MMC (custom PSD)", linewidth=2.0, color="#1f77b4")
    axes[0].loglog(freq[1:], psd_curves["white"][1:], label="White", linewidth=2.0, color="#ff7f0e")
    axes[0].loglog(freq[1:], psd_curves["pink"][1:], label="Pink", linewidth=2.0, color="#d62728")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("One-sided PSD")
    axes[0].set_title("PSD Shape")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(x_axis, mmc_noise, label="MMC noise", alpha=0.5, linewidth=1.1, color="#1f77b4")
    axes[1].plot(x_axis, white_noise, label="White noise", alpha=0.5, linewidth=1.1, color="#ff7f0e")
    axes[1].plot(x_axis, pink_noise, label="Pink noise", alpha=0.5, linewidth=1.1, color="#d62728")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Noise Traces")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outdir / "slide_1_noise_definition.png", dpi=220)
    plt.close(fig)


def write_manifest(outdir: Path, args: argparse.Namespace, cfg: dict) -> None:
    manifest = "\n".join(
        [
            "Generated by: wk5/export_noise_slide_assets.py",
            f"config: {args.config}",
            f"pair_file: {args.pair_file}",
            f"event_idx: {args.event_idx}",
            f"channel_idx: {args.channel_idx}",
            f"seed: {args.seed}",
            f"psd_draws: {args.psd_draws}",
            f"plot_start: {args.plot_start}",
            f"plot_samples: {args.plot_samples}",
            f"sampling_frequency: {cfg['sampling_frequency']}",
            f"trace_samples: {cfg['trace_samples']}",
            "",
            "Files:",
            "  slide_psd_shape.png",
            "  slide_noise_only.png",
            "  slide_trace_overlay.png",
            "  slide_1_noise_definition.png",
            "  psd_shape_data.csv",
            "  noise_only_data.csv",
            "  trace_overlay_data.csv",
            "  noise_summary_stats.csv",
            "  noise_slide_assets.npz",
        ]
    )
    (outdir / "manifest.txt").write_text(manifest, encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text())
    fs = float(cfg["sampling_frequency"])
    n_samples = int(cfg["trace_samples"])

    args.outdir.mkdir(parents=True, exist_ok=True)

    clean, mmc = load_pair_trace(args.pair_file, args.event_idx, args.channel_idx)
    if clean.shape[0] != n_samples or mmc.shape[0] != n_samples:
        raise ValueError(
            f"Trace length mismatch. config={n_samples}, clean={clean.shape[0]}, mmc={mmc.shape[0]}"
        )

    white_gen = NoiseGenerator({**cfg, "noise_type": "white"})
    pink_gen = NoiseGenerator({**cfg, "noise_type": "pink"})
    white_noise = white_gen.generate_noise(n_samples).astype(np.float32, copy=False)
    pink_noise = pink_gen.generate_noise(n_samples).astype(np.float32, copy=False)
    mmc_noise = (mmc - clean).astype(np.float32, copy=False)

    white_trace = clean + white_noise
    pink_trace = clean + pink_noise

    mmc_batch = generate_noise_batch(cfg, str(cfg["noise_type"]), n_samples=n_samples, n_draws=args.psd_draws)
    white_batch = generate_noise_batch(cfg, "white", n_samples=n_samples, n_draws=args.psd_draws)
    pink_batch = generate_noise_batch(cfg, "pink", n_samples=n_samples, n_draws=args.psd_draws)

    freq, mmc_psd = periodogram_one_sided(mmc_batch, fs)
    _, white_psd = periodogram_one_sided(white_batch, fs)
    _, pink_psd = periodogram_one_sided(pink_batch, fs)
    psd_curves = {
        "mmc_custom": np.mean(mmc_psd, axis=0),
        "white": np.mean(white_psd, axis=0),
        "pink": np.mean(pink_psd, axis=0),
    }

    sample_idx = np.arange(n_samples, dtype=np.int64)
    save_data_tables(
        args.outdir,
        sample_idx,
        freq,
        psd_curves,
        clean,
        mmc,
        white_trace,
        pink_trace,
        mmc_noise,
        white_noise,
        pink_noise,
    )

    start = max(0, int(args.plot_start))
    end = min(n_samples, start + int(args.plot_samples))
    window = slice(start, end)
    x_axis = sample_idx[window]

    plot_psd_shape(args.outdir, freq, psd_curves)
    plot_noise_only(args.outdir, x_axis, mmc_noise[window], white_noise[window], pink_noise[window])
    plot_trace_overlay(args.outdir, x_axis, clean[window], mmc[window], white_trace[window], pink_trace[window])
    plot_slide_1_combo(
        args.outdir,
        freq,
        psd_curves,
        x_axis,
        mmc_noise[window],
        white_noise[window],
        pink_noise[window],
    )

    write_manifest(args.outdir, args, cfg)
    print(f"Saved slide assets to: {args.outdir}")


if __name__ == "__main__":
    main()
