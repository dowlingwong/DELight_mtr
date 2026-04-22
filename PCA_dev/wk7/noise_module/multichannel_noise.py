"""Synthetic multi-channel noise generation built on the single-channel core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

try:
    from .NoiseGenerator import NoiseGenerator
    from .utils import match_target_std, mean_offdiag_corrcoef, resolve_rng, sample_range, spawn_rng
except ImportError:  # pragma: no cover - script execution fallback
    from NoiseGenerator import NoiseGenerator
    from utils import match_target_std, mean_offdiag_corrcoef, resolve_rng, sample_range, spawn_rng


class MultiChannelNoiseGenerator:
    """Generate independent or synthetically correlated multichannel noise."""

    DEFAULT_CONFIG = {
        "mode": "shared_private",
        "n_channels": 56,
        "corr_strength": 0.3,
        "channel_gain_jitter": 0.05,
        "n_latent": 2,
        "latent_strength_range": [0.1, 0.4],
        "private_strength_range": [0.8, 1.2],
        "normalize_channel_variance": True,
    }

    def __init__(
        self,
        base_config: dict[str, Any],
        config: dict[str, Any] | None = None,
        rng: Any = None,
        seed: int | None = None,
    ):
        self.base_config = deepcopy(base_config)
        self.config = deepcopy(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)

    def generate_independent(
        self,
        C: int,
        N: int,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate C independent channels from the base single-channel model."""
        channels = [self._make_base_generator().generate_noise(N) for _ in range(C)]
        X = np.vstack(channels)
        if self.config.get("normalize_channel_variance", True):
            X = self._normalize_channels(X)

        if return_metadata:
            return X, {
                "mode": "independent",
                "n_channels": C,
                "mean_offdiag_corr": mean_offdiag_corrcoef(X),
            }
        return X

    def generate_shared_private(
        self,
        C: int,
        N: int,
        corr_strength: float = 0.3,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate one shared latent process plus per-channel private noise."""
        corr_strength = float(np.clip(corr_strength, 0.0, 0.999))
        shared = self._make_base_generator().generate_noise(N)
        gains = 1.0 + self.rng.normal(0.0, self.config.get("channel_gain_jitter", 0.05), size=C)
        private_strengths = sample_range(
            self.rng,
            self.config.get("private_strength_range", [0.8, 1.2]),
            size=C,
        )

        X = np.empty((C, N), dtype=float)
        for idx in range(C):
            private = self._make_base_generator().generate_noise(N)
            shared_weight = gains[idx] * np.sqrt(corr_strength)
            private_weight = private_strengths[idx] * np.sqrt(max(1.0 - corr_strength, 0.0))
            X[idx] = shared_weight * shared + private_weight * private

        if self.config.get("normalize_channel_variance", True):
            X = self._normalize_channels(X)

        if return_metadata:
            return X, {
                "mode": "shared_private",
                "n_channels": C,
                "corr_strength": corr_strength,
                "mean_offdiag_corr": mean_offdiag_corrcoef(X),
            }
        return X

    def generate_lowrank_correlated(
        self,
        C: int,
        N: int,
        n_latent: int = 2,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate channels from a low-rank latent colored-process model."""
        n_latent = max(int(n_latent), 1)
        latent = np.vstack([self._make_base_generator().generate_noise(N) for _ in range(n_latent)])
        weights = self.rng.normal(0.0, 1.0, size=(C, n_latent))
        latent_strengths = sample_range(
            self.rng,
            self.config.get("latent_strength_range", [0.1, 0.4]),
            size=(C, n_latent),
        )
        private_strengths = sample_range(
            self.rng,
            self.config.get("private_strength_range", [0.8, 1.2]),
            size=C,
        )

        X = np.empty((C, N), dtype=float)
        for idx in range(C):
            private = self._make_base_generator().generate_noise(N)
            shared = np.sum(weights[idx, :, None] * latent_strengths[idx, :, None] * latent, axis=0)
            X[idx] = shared + private_strengths[idx] * private

        if self.config.get("normalize_channel_variance", True):
            X = self._normalize_channels(X)

        if return_metadata:
            return X, {
                "mode": "lowrank",
                "n_channels": C,
                "n_latent": n_latent,
                "mean_offdiag_corr": mean_offdiag_corrcoef(X),
            }
        return X

    def generate(
        self,
        N: int,
        C: int | None = None,
        mode: str | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate according to the configured multichannel mode."""
        C = int(C or self.config.get("n_channels", 1))
        mode = (mode or self.config.get("mode", "shared_private")).lower()

        if mode == "independent":
            return self.generate_independent(C, N, return_metadata=return_metadata)
        if mode == "shared_private":
            return self.generate_shared_private(
                C,
                N,
                corr_strength=float(self.config.get("corr_strength", 0.3)),
                return_metadata=return_metadata,
            )
        if mode == "lowrank":
            return self.generate_lowrank_correlated(
                C,
                N,
                n_latent=int(self.config.get("n_latent", 2)),
                return_metadata=return_metadata,
            )
        raise ValueError(f"Unsupported multichannel mode: {mode}")

    def _make_base_generator(self) -> NoiseGenerator:
        return NoiseGenerator(self.base_config, rng=spawn_rng(self.rng))

    def _normalize_channels(self, X: np.ndarray) -> np.ndarray:
        target_std = np.sqrt(float(self.base_config.get("noise_power", 1.0)))
        return match_target_std(X, target_std=target_std, axis=1)
