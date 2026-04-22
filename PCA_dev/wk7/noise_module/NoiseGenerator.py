"""Single-channel stationary spectral noise generator."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import numpy as np

from scipy.fft import irfft, rfftfreq
from scipy.interpolate import interp1d

try:
    from .utils import resolve_rng
except ImportError:  # pragma: no cover - script execution fallback
    from utils import resolve_rng


class NoiseGenerator:
    """Generate stationary single-channel Gaussian noise from a target PSD."""

    def __init__(self, config: dict[str, Any], rng: Any = None, seed: int | None = None):
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)
        self.config: dict[str, Any] = {}
        self._noise_path: str | None = None
        self._set_spectra()
        self.set_config(config)

    def set_config(self, config: dict[str, Any]) -> None:
        """Validate and store a new generator configuration."""
        expected_fields = ["noise_type", "noise_power", "sampling_frequency"]
        missing = [field for field in expected_fields if field not in config]
        if missing:
            raise RuntimeError(
                f"Configuration missing required field(s): {', '.join(missing)}."
            )

        self.config = deepcopy(config)
        self.set_noise_type(config["noise_type"])
        self.set_noise_power(config["noise_power"])
        self.sampling_frequency = float(config["sampling_frequency"])

    def set_noise_type(self, noise_type: str) -> None:
        """Set the analytic noise colour or load a custom PSD file."""
        analytic_types = "white blue violet brownian pink".split()
        if isinstance(noise_type, str) and noise_type.lower() in analytic_types:
            self.noise_type = noise_type.lower()
            self._noise_path = None
        elif os.path.isfile(str(noise_type)):
            self._noise_path = os.path.abspath(str(noise_type))
            self.noise_type = "custom"
            self._load_psd()
        else:
            raise RuntimeError(
                f"Configuration noise_type field {noise_type} is neither a supported "
                "noise type nor a PSD file path."
            )
        self.spectrum = self._spectra[self.noise_type]

    def set_noise_power(self, noise_power: float) -> None:
        """Set the target integrated noise power."""
        self.psd_area = float(noise_power)

    def _set_spectra(self) -> None:
        self._spectra = {
            "white": lambda f: np.ones_like(f, dtype=float),
            "blue": lambda f: f,
            "violet": lambda f: f**2,
            "brownian": lambda f: 1.0 / np.where(f == 0, np.inf, f**2),
            "pink": lambda f: 1.0 / np.where(f == 0, np.inf, f),
        }
        self._normalize = {
            "white": self._normalize_white,
            "blue": self._normalize_blue,
            "violet": self._normalize_violet,
            "brownian": self._normalize_brownian,
            "pink": self._normalize_pink,
        }

    def _load_psd(self) -> None:
        if self._noise_path is None:
            raise RuntimeError("Custom PSD path is not set.")
        self.noise_psd_data = np.load(self._noise_path)
        self._spectra["custom"] = interp1d(
            self.noise_psd_data[0],
            self.noise_psd_data[1],
            bounds_error=False,
            fill_value=(self.noise_psd_data[1][0], self.noise_psd_data[1][-1]),
        )
        self._normalize["custom"] = lambda f: np.ones_like(f, dtype=float)

    def build_psd(
        self, N: int, return_metadata: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return the one-sided PSD used for stationary synthesis."""
        if N <= 0:
            raise ValueError("N must be positive.")

        frequencies = rfftfreq(N, d=1.0 / self.sampling_frequency)
        norm = (
            0.5
            * self.psd_area
            * self._normalize[self.noise_type](frequencies)
            * self.sampling_frequency
            * N
        )
        psd = np.asarray(norm * self.spectrum(frequencies), dtype=float)
        psd = np.clip(psd, a_min=0.0, a_max=None)

        if self.noise_type == "custom":
            psd = psd / (0.5 * self.psd_area)
            upper = N // 2 + 1 - (N + 1) % 2
            psd[1:upper] *= 0.5

        if not return_metadata:
            return frequencies, psd

        metadata = {
            "noise_type": self.noise_type,
            "noise_power": self.psd_area,
            "sampling_frequency": self.sampling_frequency,
            "seed": self.seed,
            "n_samples": N,
            "psd_total": float(np.sum(psd)),
            "config": deepcopy(self.config),
        }
        return frequencies, psd, metadata

    def sample_stationary_gaussian_from_psd(
        self,
        psd: np.ndarray,
        N: int | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Sample a stationary Gaussian time series from a one-sided PSD."""
        psd = np.asarray(psd, dtype=float)
        if psd.ndim != 1:
            raise ValueError("PSD must be a one-dimensional array.")
        if N is None:
            N = 2 * (len(psd) - 1)
        if N <= 0:
            raise ValueError("N must be positive.")

        amplitude = np.sqrt(np.clip(psd, a_min=0.0, a_max=None))
        spectrum = np.zeros_like(amplitude, dtype=complex)

        if len(amplitude) > 0:
            spectrum[0] = amplitude[0]
        if len(amplitude) > 2:
            phases = self.rng.uniform(0.0, 2.0 * np.pi, len(amplitude) - 2)
            spectrum[1:-1] = amplitude[1:-1] * np.exp(1j * phases)
        if len(amplitude) > 1:
            spectrum[-1] = amplitude[-1] if N % 2 == 0 else amplitude[-1] * np.exp(
                1j * self.rng.uniform(0.0, 2.0 * np.pi)
            )

        signal = irfft(spectrum, n=N)
        if not return_metadata:
            return signal

        metadata = {
            "n_samples": N,
            "variance": float(np.var(signal)),
            "mean": float(np.mean(signal)),
        }
        return signal, metadata

    def generate_noise(
        self, N: int, return_metadata: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Convenience wrapper combining PSD construction and sampling."""
        if return_metadata:
            frequencies, psd, psd_meta = self.build_psd(N, return_metadata=True)
            signal, sample_meta = self.sample_stationary_gaussian_from_psd(
                psd,
                N=N,
                return_metadata=True,
            )
            metadata = {
                **psd_meta,
                **sample_meta,
                "frequencies": frequencies,
            }
            return signal, metadata

        _, psd = self.build_psd(N)
        return self.sample_stationary_gaussian_from_psd(psd, N=N)

    @staticmethod
    def _normalize_white(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        return 1.0 / max(float(np.max(frequencies) - np.min(frequencies)), np.finfo(float).eps)

    @staticmethod
    def _normalize_blue(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        denom = float(np.max(frequencies) ** 2 - np.min(frequencies) ** 2)
        return 2.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_violet(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        denom = float(np.max(frequencies) ** 3 - np.min(frequencies) ** 3)
        return 3.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_brownian(frequencies: np.ndarray) -> float:
        positive = np.sort(frequencies[frequencies > 0])
        if len(positive) == 0:
            return 1.0
        if len(positive) == 1:
            return float(positive[0])
        denom = float(1.0 / positive[0] - 1.0 / positive[-1])
        return 1.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_pink(frequencies: np.ndarray) -> float:
        positive = np.sort(frequencies[frequencies > 0])
        if len(positive) < 2:
            return 1.0
        denom = float(np.log(positive[-1]) - np.log(positive[0]))
        return 1.0 / max(denom, np.finfo(float).eps)
