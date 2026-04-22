"""Artifact injection on top of baseline noise traces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

try:
    from .templates import generate_burst_template, generate_glitch_template
    from .utils import resolve_rng, sample_range, spawn_rng
except ImportError:  # pragma: no cover - script execution fallback
    from templates import generate_burst_template, generate_glitch_template
    from utils import resolve_rng, sample_range, spawn_rng


class ArtifactInjector:
    """Inject deterministic lines and transient artifacts into traces."""

    DEFAULT_CONFIG = {
        "sampling_frequency": 1.0,
        "enable_lines": False,
        "lines": [],
        "enable_glitches": False,
        "glitch_rate": 1.0,
        "glitch_amp_range": [0.05, 0.2],
        "glitch_templates": ["impulse", "exp_decay", "damped_sine"],
        "glitch_duration_samples": [32, 256],
        "enable_bursts": False,
        "burst_rate": 0.2,
        "burst_amp_range": [0.03, 0.1],
        "burst_duration_samples": [128, 512],
        "enable_sparse_impulses": False,
        "impulse_probability": 1e-4,
        "impulse_sigma": 0.1,
        "channel_amplitude_jitter": 0.05,
    }

    def __init__(self, config: dict[str, Any] | None = None, rng: Any = None, seed: int | None = None):
        self.config = deepcopy(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)

    def apply(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply configured artifacts to a single-channel trace."""
        y = np.array(x, dtype=float, copy=True)
        metadata: dict[str, Any] = {}

        if self.config.get("enable_lines", False):
            y, line_meta = self.add_lines(y, return_metadata=True)
            metadata["lines"] = line_meta
        if self.config.get("enable_glitches", False):
            y, glitch_meta = self.add_glitches(y, return_metadata=True)
            metadata["glitches"] = glitch_meta
        if self.config.get("enable_bursts", False):
            y, burst_meta = self.add_bursts(y, return_metadata=True)
            metadata["bursts"] = burst_meta
        if self.config.get("enable_sparse_impulses", False):
            y, impulse_meta = self.add_sparse_impulses(y, return_metadata=True)
            metadata["sparse_impulses"] = impulse_meta

        if return_metadata:
            metadata["output_std"] = float(np.std(y))
            return y, metadata
        return y

    def apply_multichannel(
        self,
        X: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply artifacts to a multichannel array of shape (C, N)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Multichannel input must have shape (C, N).")

        C, _ = X.shape
        output_channels = []
        metadata_channels = []
        for idx in range(C):
            channel_rng = spawn_rng(self.rng)
            channel_injector = ArtifactInjector(self.config, rng=channel_rng)
            jitter = 1.0 + self.rng.normal(0.0, self.config.get("channel_amplitude_jitter", 0.05))
            channel, channel_meta = channel_injector.apply(X[idx] * jitter, return_metadata=True)
            output_channels.append(channel)
            metadata_channels.append(channel_meta)

        Y = np.vstack(output_channels)
        if return_metadata:
            return Y, {"channels": metadata_channels}
        return Y

    def add_lines(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add deterministic sinusoidal spectral lines."""
        y = np.array(x, dtype=float, copy=True)
        fs = float(self.config.get("sampling_frequency", 1.0))
        t = np.arange(len(y), dtype=float) / max(fs, 1.0)
        added_lines = []

        for line_cfg in self.config.get("lines", []):
            base_freq = float(line_cfg["freq"])
            base_amp = line_cfg["amp"]
            phase_cfg = line_cfg.get("phase", "random")
            harmonics = line_cfg.get("harmonics", [1])
            for harmonic in harmonics:
                freq = base_freq * float(harmonic)
                amp = sample_range(self.rng, base_amp)
                phase = (
                    self.rng.uniform(0.0, 2.0 * np.pi)
                    if phase_cfg == "random"
                    else float(phase_cfg)
                )
                y += amp * np.sin(2.0 * np.pi * freq * t + phase)
                added_lines.append({"freq": freq, "amp": float(amp)})

        if return_metadata:
            return y, {"count": len(added_lines), "lines": added_lines}
        return y

    def add_glitches(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add localized transient glitch templates."""
        y = np.array(x, dtype=float, copy=True)
        fs = float(self.config.get("sampling_frequency", 1.0))
        glitch_rate = float(self.config.get("glitch_rate", 0.0))
        n_glitches = int(self.rng.poisson(glitch_rate))
        injected = []

        for _ in range(n_glitches):
            duration = int(round(sample_range(self.rng, self.config["glitch_duration_samples"])))
            duration = max(duration, 4)
            if duration >= len(y):
                duration = max(len(y) - 1, 4)
            if duration <= 0 or duration >= len(y):
                continue
            start = int(self.rng.integers(0, len(y) - duration))
            amp = float(sample_range(self.rng, self.config["glitch_amp_range"]))
            kind = str(self.rng.choice(self.config["glitch_templates"]))
            template = generate_glitch_template(kind, duration, fs, self.rng)
            y[start : start + duration] += amp * template
            injected.append({"start": start, "duration": duration, "amp": amp, "kind": kind})

        if return_metadata:
            return y, {"count": len(injected), "glitches": injected}
        return y

    def add_bursts(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add short bursty packets."""
        y = np.array(x, dtype=float, copy=True)
        fs = float(self.config.get("sampling_frequency", 1.0))
        burst_rate = float(self.config.get("burst_rate", 0.0))
        n_bursts = int(self.rng.poisson(burst_rate))
        injected = []

        for _ in range(n_bursts):
            duration = int(round(sample_range(self.rng, self.config["burst_duration_samples"])))
            duration = max(duration, 8)
            if duration >= len(y):
                duration = max(len(y) - 1, 8)
            if duration <= 0 or duration >= len(y):
                continue
            start = int(self.rng.integers(0, len(y) - duration))
            amp = float(sample_range(self.rng, self.config["burst_amp_range"]))
            burst = generate_burst_template(duration, fs, self.rng)
            y[start : start + duration] += amp * burst
            injected.append({"start": start, "duration": duration, "amp": amp})

        if return_metadata:
            return y, {"count": len(injected), "bursts": injected}
        return y

    def add_sparse_impulses(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add sparse heavy-tail outliers."""
        y = np.array(x, dtype=float, copy=True)
        probability = float(self.config.get("impulse_probability", 0.0))
        sigma = float(self.config.get("impulse_sigma", 0.0))
        mask = self.rng.uniform(0.0, 1.0, size=len(y)) < probability
        impulses = self.rng.normal(0.0, sigma, size=int(np.sum(mask)))
        y[mask] += impulses

        if return_metadata:
            return y, {"count": int(np.sum(mask)), "sigma": sigma}
        return y
