from __future__ import annotations

import torch

from actlearn_gsm.config import EnsembleConfig, GridConfig
from actlearn_gsm.data import generate_geospatial_grid
from actlearn_gsm.ensemble import GSMEnsemble


def test_generate_geospatial_grid_shapes() -> None:
    dataset = generate_geospatial_grid(GridConfig(n_lon=5, n_lat=4, seed=7))

    assert dataset.X_all.shape == (20, 2)
    assert dataset.y_clean.shape == (20,)
    assert dataset.y_noisy.shape == (20,)
    assert torch.isfinite(dataset.y_noisy).all()


def test_gsm_ensemble_scores_tiny_batch() -> None:
    dataset = generate_geospatial_grid(GridConfig(n_lon=5, n_lat=4, seed=7))
    ensemble = GSMEnsemble(
        EnsembleConfig(
            size=2,
            num_inducing=3,
            num_mixtures=2,
            map_iters=1,
            lr=0.01,
        )
    )

    ensemble.build(dataset.X_all[:6], dataset.y_noisy[:6], logger=None)
    scores = ensemble.mc_epistemic_variance(dataset.X_all[:3])

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
