"""Temporal wrappers for non-stationary extensions of the base noise model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from scipy.interpolate import CubicSpline

try:
    from .NoiseGenerator import NoiseGenerator
    from .utils import concatenate_with_crossfade, resolve_rng, sample_range, spawn_rng
except ImportError:  # pragma: no cover - script execution fallback
    from NoiseGenerator import NoiseGenerator
    from utils import concatenate_with_crossfade, resolve_rng, sample_range, spawn_rng


class TemporalNoiseWrapper:
    """Apply piecewise stationarity, drift, and slow variance changes."""

    DEFAULT_CONFIG = {
        "mode": "none",
        "n_segments": 4,
        "segment_length": None,
        "crossfade_len": 128,
        "vary_noise_power": True,
        "noise_power_scale_range": [0.8, 1.2],
        "vary_psd_slope": False,
        "psd_slope_range": [-0.1, 0.1],
        "add_drift": False,
        "drift_type": "spline",
        "drift_sigma": 0.05,
        "drift_n_knots": 6,
        "variance_modulation": False,
        "variance_scale_range": [0.95, 1.05],
        "variance_n_knots": 6,
        "multichannel_shared_drift": True,
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
        base_generator: NoiseGenerator | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply temporal effects to a single-channel trace."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("Single-channel input must be one-dimensional.")

        metadata: dict[str, Any] = {"mode": self.config["mode"]}
        y = np.array(x, copy=True)

        if self.config["mode"] == "piecewise":
            if base_generator is not None:
                y, piecewise_meta = self.generate_piecewise(
                    len(x),
                    base_generator=base_generator,
                    return_metadata=True,
                )
            else:
                y, piecewise_meta = self._apply_piecewise_scaling(y, return_metadata=True)
            metadata["piecewise"] = piecewise_meta

        if self.config.get("variance_modulation", False):
            envelope = self._build_variance_envelope(len(y))
            y = y * envelope
            metadata["variance_envelope_range"] = [
                float(np.min(envelope)),
                float(np.max(envelope)),
            ]

        if self.config.get("add_drift", False):
            drift = self.generate_drift(len(y))
            y = y + drift
            metadata["drift_std"] = float(np.std(drift))

        if return_metadata:
            metadata["output_std"] = float(np.std(y))
            return y, metadata
        return y

    def apply_multichannel(
        self,
        X: np.ndarray,
        base_generator: Any = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply temporal effects to a multichannel array of shape (C, N)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Multichannel input must have shape (C, N).")

        C, N = X.shape
        Y = np.array(X, copy=True)
        metadata: dict[str, Any] = {"mode": self.config["mode"], "n_channels": C}

        if self.config["mode"] == "piecewise":
            if base_generator is not None:
                Y, piecewise_meta = self._generate_piecewise_array(
                    N,
                    base_generator=base_generator,
                    return_metadata=True,
                )
            else:
                scaled_channels = []
                channel_meta = []
                for channel in Y:
                    channel_rng = spawn_rng(self.rng)
                    channel_wrapper = TemporalNoiseWrapper(self.config, rng=channel_rng)
                    scaled, channel_piecewise = channel_wrapper._apply_piecewise_scaling(
                        channel,
                        return_metadata=True,
                    )
                    scaled_channels.append(scaled)
                    channel_meta.append(channel_piecewise)
                Y = np.vstack(scaled_channels)
                piecewise_meta = {"channels": channel_meta}
            metadata["piecewise"] = piecewise_meta

        if self.config.get("variance_modulation", False):
            shared_envelope = self._build_variance_envelope(N)
            Y = Y * shared_envelope[None, :]
            metadata["variance_envelope_range"] = [
                float(np.min(shared_envelope)),
                float(np.max(shared_envelope)),
            ]

        if self.config.get("add_drift", False):
            if self.config.get("multichannel_shared_drift", True):
                drift = self.generate_drift(N)
                drift_scales = sample_range(self.rng, [0.9, 1.1], size=C)
                Y = Y + drift_scales[:, None] * drift[None, :]
            else:
                drift = np.vstack([self.generate_drift(N) for _ in range(C)])
                Y = Y + drift
            metadata["drift_std"] = float(np.std(drift))

        if return_metadata:
            metadata["output_std"] = float(np.std(Y))
            return Y, metadata
        return Y

    def generate_piecewise(
        self,
        N: int,
        base_generator: NoiseGenerator,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate a piecewise-stationary single-channel trace from scratch."""
        signal, metadata = self._generate_piecewise_array(
            N,
            base_generator=base_generator,
            return_metadata=True,
        )
        if signal.ndim != 1:
            raise RuntimeError("Expected single-channel output from piecewise generation.")
        if return_metadata:
            return signal, metadata
        return signal

    def generate_drift(self, N: int) -> np.ndarray:
        """Generate a smooth additive low-frequency drift component."""
        if N <= 0:
            raise ValueError("N must be positive.")
        drift_type = self.config.get("drift_type", "spline")
        sigma = float(self.config.get("drift_sigma", 0.05))
        if sigma == 0:
            return np.zeros(N, dtype=float)

        if drift_type == "random_walk":
            steps = self.rng.normal(0.0, sigma / max(np.sqrt(N), 1.0), size=N)
            drift = np.cumsum(steps)
            drift = drift - np.mean(drift)
            return drift

        n_knots = max(int(self.config.get("drift_n_knots", 6)), 2)
        knot_x = np.linspace(0.0, N - 1, n_knots)
        knot_y = self.rng.normal(0.0, sigma, size=n_knots)
        if n_knots >= 4:
            interpolator = CubicSpline(knot_x, knot_y, bc_type="natural")
            drift = interpolator(np.arange(N, dtype=float))
        else:
            drift = np.interp(np.arange(N, dtype=float), knot_x, knot_y)
        drift = drift - np.mean(drift)
        return drift

    def _generate_piecewise_array(
        self,
        N: int,
        base_generator: Any,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        segments = self._build_segments(N)
        crossfade_len = max(int(self.config.get("crossfade_len", 0)), 0)
        outputs = []
        metadata_segments = []

        for idx, (start, end) in enumerate(segments):
            seg_len = end - start
            effective_len = seg_len + (crossfade_len if idx > 0 else 0)
            local_cfg = self._sample_local_config(self._extract_base_config(base_generator))
            local_output = self._generate_local_segment(
                base_generator=base_generator,
                local_config=local_cfg,
                seg_len=effective_len,
            )
            outputs.append(local_output)
            metadata_segments.append(
                {
                    "start": start,
                    "end": end,
                    "noise_power": float(local_cfg["noise_power"]),
                    "noise_type": local_cfg["noise_type"],
                }
            )

        combined = concatenate_with_crossfade(outputs, crossfade_len)
        if combined.shape[-1] != N:
            combined = combined[..., :N]

        if return_metadata:
            metadata = {
                "segments": metadata_segments,
                "crossfade_len": int(self.config.get("crossfade_len", 0)),
            }
            return combined, metadata
        return combined

    def _apply_piecewise_scaling(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        segments = self._build_segments(len(x))
        crossfade_len = max(int(self.config.get("crossfade_len", 0)), 0)
        outputs = []
        metadata_segments = []
        base_power = float(np.var(x)) if np.var(x) > 0 else 1.0

        for idx, (start, end) in enumerate(segments):
            scale = 1.0
            if self.config.get("vary_noise_power", True):
                power_scale = sample_range(
                    self.rng,
                    self.config.get("noise_power_scale_range", [0.8, 1.2]),
                )
                scale = float(np.sqrt(max(power_scale, 0.0)))
            segment_start = max(0, start - crossfade_len) if idx > 0 else start
            segment = np.array(x[segment_start:end], copy=True) * scale
            outputs.append(segment)
            metadata_segments.append(
                {
                    "start": start,
                    "end": end,
                    "approx_noise_power": base_power * scale**2,
                }
            )

        combined = concatenate_with_crossfade(outputs, crossfade_len)
        combined = combined[: len(x)]
        if return_metadata:
            return combined, {"segments": metadata_segments}
        return combined

    def _build_segments(self, N: int) -> list[tuple[int, int]]:
        if N <= 0:
            raise ValueError("N must be positive.")
        segment_length = self.config.get("segment_length")
        if segment_length is None:
            n_segments = max(int(self.config.get("n_segments", 1)), 1)
            segment_edges = np.linspace(0, N, n_segments + 1, dtype=int)
            return [
                (int(segment_edges[i]), int(segment_edges[i + 1]))
                for i in range(n_segments)
                if int(segment_edges[i + 1]) > int(segment_edges[i])
            ]

        segment_length = max(int(segment_length), 1)
        starts = np.arange(0, N, segment_length, dtype=int)
        ends = np.minimum(starts + segment_length, N)
        return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]

    def _sample_local_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        local_config = deepcopy(base_config)
        if self.config.get("vary_noise_power", True):
            scale = sample_range(
                self.rng,
                self.config.get("noise_power_scale_range", [0.8, 1.2]),
            )
            local_config["noise_power"] = float(base_config["noise_power"]) * float(scale)
        return local_config

    @staticmethod
    def _extract_base_config(base_generator: Any) -> dict[str, Any]:
        if hasattr(base_generator, "base_config"):
            return deepcopy(base_generator.base_config)
        if hasattr(base_generator, "config"):
            return deepcopy(base_generator.config)
        raise ValueError("base_generator does not expose a usable configuration.")

    def _generate_local_segment(
        self,
        base_generator: Any,
        local_config: dict[str, Any],
        seg_len: int,
    ) -> np.ndarray:
        local_rng = spawn_rng(self.rng)
        if hasattr(base_generator, "generate_noise"):
            local_generator = base_generator.__class__(local_config, rng=local_rng)
            return np.asarray(local_generator.generate_noise(seg_len), dtype=float)

        if hasattr(base_generator, "generate"):
            local_generator = base_generator.__class__(
                local_config,
                config=getattr(base_generator, "config", None),
                rng=local_rng,
            )
            return np.asarray(local_generator.generate(seg_len), dtype=float)

        raise ValueError("Unsupported base_generator type for piecewise generation.")

    def _build_variance_envelope(self, N: int) -> np.ndarray:
        n_knots = max(int(self.config.get("variance_n_knots", 6)), 2)
        knot_x = np.linspace(0.0, N - 1, n_knots)
        knot_y = sample_range(
            self.rng,
            self.config.get("variance_scale_range", [0.95, 1.05]),
            size=n_knots,
        )
        if n_knots >= 4:
            interpolator = CubicSpline(knot_x, knot_y, bc_type="natural")
            envelope = interpolator(np.arange(N, dtype=float))
        else:
            envelope = np.interp(np.arange(N, dtype=float), knot_x, knot_y)
        return np.clip(envelope, 1e-3, None)
