"""Composable noise generation modules for wk7 experiments."""

from .NoiseGenerator import NoiseGenerator
from .artifact_injector import ArtifactInjector
from .multichannel_noise import MultiChannelNoiseGenerator
from .temporal_noise import TemporalNoiseWrapper

__all__ = [
    "ArtifactInjector",
    "MultiChannelNoiseGenerator",
    "NoiseGenerator",
    "TemporalNoiseWrapper",
]
