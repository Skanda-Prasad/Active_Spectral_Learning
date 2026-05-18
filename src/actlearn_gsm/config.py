"""Experiment configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

Acquisition = Literal["parameter", "prediction"]


@dataclass(frozen=True)
class GridConfig:
    n_lon: int = 35
    n_lat: int = 45
    seed: int = 42


@dataclass(frozen=True)
class EnsembleConfig:
    size: int = 6
    num_inducing: int = 8
    num_mixtures: int = 5
    map_iters: int = 150
    lr: float = 0.03


@dataclass(frozen=True)
class BaselineConfig:
    num_mixtures: int = 5
    iters: int = 150
    lr: float = 0.05


@dataclass(frozen=True)
class ActiveLearningConfig:
    num_init: int = 100
    num_steps: int = 8
    plot_every: int = 2
    seed: int = 123
    batch_size: int = 512


@dataclass(frozen=True)
class ExperimentConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    artifact_dir: Path = Path("artifacts")
    make_plots: bool = True
    device: str | None = None


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a user-requested device name to a torch device."""
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
