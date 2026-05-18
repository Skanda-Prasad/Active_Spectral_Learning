"""Experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from actlearn_gsm.active_learning import ActiveLearner, ActiveLearningResults
from actlearn_gsm.config import ExperimentConfig, resolve_device
from actlearn_gsm.data import generate_geospatial_grid


@dataclass(frozen=True)
class ExperimentResults:
    parameter: ActiveLearningResults
    prediction: ActiveLearningResults


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    logger=print,
) -> ExperimentResults:
    """Run the full MC-EV vs predictive-variance comparison."""
    config = config or ExperimentConfig()
    torch.set_default_dtype(torch.float32)
    device = resolve_device(config.device)

    artifact_dir = config.artifact_dir
    figure_dir = artifact_dir / "figures"
    if config.make_plots:
        figure_dir.mkdir(parents=True, exist_ok=True)

    _log(logger, "=" * 70)
    _log(logger, "PARAMETER-INFORMATION ACTIVE LEARNING: MC-EV VS PRED VAR")
    _log(logger, "=" * 70)
    _log(logger, f"Using device: {device}")
    _log(logger, "Generating 2D geospatial-like dataset...")

    dataset = generate_geospatial_grid(config.grid, device=device)
    _log(
        logger,
        (
            f"Full grid: {dataset.n_lat} x {dataset.n_lon} = "
            f"{dataset.size} points in [0, 1]^2"
        ),
    )

    _log(logger, "")
    _log(logger, "=" * 70)
    _log(logger, "RUNNING: PARAMETER INFORMATION ACQUISITION (MC-EV)")
    _log(logger, "=" * 70)
    parameter_learner = ActiveLearner(
        dataset,
        acquisition="parameter",
        ensemble_config=config.ensemble,
        baseline_config=config.baseline,
        figure_dir=figure_dir,
        make_plots=config.make_plots,
        logger=logger,
    )
    parameter_results = parameter_learner.run(config.active_learning)

    _log(logger, "")
    _log(logger, "=" * 70)
    _log(logger, "RUNNING: PREDICTIVE VARIANCE ACQUISITION")
    _log(logger, "=" * 70)
    prediction_learner = ActiveLearner(
        dataset,
        acquisition="prediction",
        ensemble_config=config.ensemble,
        baseline_config=config.baseline,
        figure_dir=figure_dir,
        make_plots=config.make_plots,
        logger=logger,
    )
    prediction_results = prediction_learner.run(config.active_learning)

    if config.make_plots:
        from actlearn_gsm.plotting import plot_learning_curves_comparison

        _log(logger, "")
        _log(logger, "=" * 70)
        _log(logger, "GENERATING COMPARISON PLOT")
        _log(logger, "=" * 70)
        plot_learning_curves_comparison(
            parameter_results.as_dict(),
            prediction_results.as_dict(),
            save_path=artifact_dir / "final_comparison_param_vs_pred.png",
        )

    _log_summary(logger, parameter_results, prediction_results)
    return ExperimentResults(parameter=parameter_results, prediction=prediction_results)


def _log(logger, message: str) -> None:
    if logger:
        logger(message)


def _log_summary(
    logger,
    parameter_results: ActiveLearningResults,
    prediction_results: ActiveLearningResults,
) -> None:
    _log(logger, "")
    _log(logger, "=" * 70)
    _log(logger, "ACTIVE LEARNING EXPERIMENT COMPLETE")
    _log(logger, "=" * 70)
    _log(logger, "Final Results:")
    _log(logger, "  Parameter-based MC-EV:")
    _log(logger, f"    - Final MSE: {parameter_results.mse_gsm[-1]:.4f}")
    _log(logger, f"    - Final trace(Cov(psi)): {parameter_results.trace_cov[-1]:.4f}")
    _log(logger, f"    - Avg MSE: {np.mean(parameter_results.mse_gsm):.4f}")
    _log(logger, "  Prediction-based variance:")
    _log(logger, f"    - Final MSE: {prediction_results.mse_gsm[-1]:.4f}")
    _log(logger, f"    - Final trace(Cov(psi)): {prediction_results.trace_cov[-1]:.4f}")
    _log(logger, f"    - Avg MSE: {np.mean(prediction_results.mse_gsm):.4f}")

    winner_mse = (
        "Parameter-based MC-EV"
        if parameter_results.mse_gsm[-1] < prediction_results.mse_gsm[-1]
        else "Prediction-based variance"
    )
    winner_trace = (
        "Parameter-based MC-EV"
        if parameter_results.trace_cov[-1] < prediction_results.trace_cov[-1]
        else "Prediction-based variance"
    )
    _log(logger, f"  Winner (Final MSE): {winner_mse}")
    _log(logger, f"  Winner (Param Uncertainty): {winner_trace}")
