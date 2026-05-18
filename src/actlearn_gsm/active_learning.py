"""Active-learning loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import gpytorch
import torch

from actlearn_gsm.config import (
    Acquisition,
    ActiveLearningConfig,
    BaselineConfig,
    EnsembleConfig,
)
from actlearn_gsm.data import GeospatialDataset
from actlearn_gsm.ensemble import GSMEnsemble
from actlearn_gsm.models import train_spectral_mixture_baseline
from actlearn_gsm.uncertainty import hyper_param_uncertainty


Logger = Callable[[str], None]


@dataclass
class ActiveLearningResults:
    iter: list[int] = field(default_factory=list)
    mse_gsm: list[float] = field(default_factory=list)
    mse_sm: list[float] = field(default_factory=list)
    trace_cov: list[float] = field(default_factory=list)
    logdet_cov: list[float] = field(default_factory=list)
    train_sizes: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "iter": self.iter,
            "mse_gsm": self.mse_gsm,
            "mse_sm": self.mse_sm,
            "trace_cov": self.trace_cov,
            "logdet_cov": self.logdet_cov,
            "train_sizes": self.train_sizes,
        }


class ActiveLearner:
    """Run one active-learning strategy on a fixed synthetic dataset."""

    def __init__(
        self,
        dataset: GeospatialDataset,
        *,
        acquisition: Acquisition,
        ensemble_config: EnsembleConfig,
        baseline_config: BaselineConfig,
        figure_dir: Path | None = None,
        make_plots: bool = True,
        logger: Logger | None = print,
    ) -> None:
        self.dataset = dataset
        self.acquisition = acquisition
        self.baseline_config = baseline_config
        self.figure_dir = figure_dir
        self.make_plots = make_plots
        self.logger = logger
        self.ensemble = GSMEnsemble(ensemble_config)
        self.results = ActiveLearningResults()

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def initialize(self, num_init: int, seed: int) -> None:
        torch.manual_seed(seed)
        permutation = torch.randperm(self.dataset.size, device=self.dataset.X_all.device)
        self.train_idx = permutation[:num_init].clone()
        self.pool_idx = permutation[num_init:].clone()
        self.train_x = self.dataset.X_all[self.train_idx].clone()
        self.train_y = self.dataset.y_noisy[self.train_idx].clone()

        self._log(f"Initial training set size: {self.train_x.shape[0]}")
        self._log(f"Acquisition function: {self.acquisition}")

    def pick_next_point(self, batch_size: int) -> tuple[int, float]:
        X_pool = self.dataset.X_all[self.pool_idx]
        if self.acquisition == "parameter":
            acquisition_values = self.ensemble.mc_epistemic_variance(
                X_pool,
                batch_size=batch_size,
            )
            acquisition_name = "MC-EV"
        elif self.acquisition == "prediction":
            acquisition_values = self.ensemble.predictive_variance_single_model(
                X_pool,
                batch_size=batch_size,
            )
            acquisition_name = "Predictive variance"
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")

        best_local = torch.argmax(acquisition_values)
        best_global = self.pool_idx[best_local]
        score = acquisition_values[best_local].item()
        self._log(
            f"  Acquired via {acquisition_name}: "
            f"idx={best_global.item()}, score={score:.4f}"
        )
        return best_global.item(), score

    def update_dataset(self, idx: int) -> None:
        self.train_x = torch.cat(
            [self.train_x, self.dataset.X_all[idx : idx + 1]],
            dim=0,
        )
        self.train_y = torch.cat(
            [self.train_y, self.dataset.y_noisy[idx : idx + 1]],
            dim=0,
        )
        self.pool_idx = self.pool_idx[self.pool_idx != idx]

    def run(self, config: ActiveLearningConfig) -> ActiveLearningResults:
        self.initialize(num_init=config.num_init, seed=config.seed)

        for iteration in range(1, config.num_steps + 1):
            self._log("")
            self._log("=" * 70)
            self._log(
                f"AL ITERATION {iteration}/{config.num_steps} "
                f"({self.acquisition.upper()})"
            )
            self._log(f"Current train size: {self.train_x.shape[0]}")
            self._log("=" * 70)

            self.ensemble.build(self.train_x, self.train_y, logger=self.logger)
            sm_model, sm_likelihood = train_spectral_mixture_baseline(
                self.train_x,
                self.train_y,
                self.baseline_config,
                logger=self.logger,
            )

            with torch.no_grad():
                ensemble_means = self.ensemble.ensemble_predict_means(
                    self.dataset.X_all,
                    batch_size=config.batch_size,
                )
                gsm_mean = ensemble_means.mean(dim=0)
                mse_gsm = torch.mean(
                    (gsm_mean - self.dataset.y_clean) ** 2
                ).item()

                with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
                    1e-3
                ):
                    sm_output = sm_likelihood(sm_model(self.dataset.X_all))
                mse_sm = torch.mean(
                    (sm_output.mean - self.dataset.y_clean) ** 2
                ).item()

            trace_psi, logdet_psi = hyper_param_uncertainty(self.ensemble)
            self._log(f"  GSM MSE: {mse_gsm:.4f}, SM MSE: {mse_sm:.4f}")
            self._log(
                f"  trace(Cov(psi)): {trace_psi:.4f}, "
                f"logdet(Cov(psi)): {logdet_psi:.4f}"
            )

            self.results.iter.append(iteration)
            self.results.mse_gsm.append(mse_gsm)
            self.results.mse_sm.append(mse_sm)
            self.results.trace_cov.append(trace_psi)
            self.results.logdet_cov.append(logdet_psi)
            self.results.train_sizes.append(self.train_x.shape[0])

            should_plot = iteration % config.plot_every == 0 or iteration == 1
            if self.make_plots and self.figure_dir and should_plot:
                from actlearn_gsm.plotting import (
                    plot_acquisition_comparison,
                    plot_hyperparameter_uncertainty_grid,
                )

                plot_hyperparameter_uncertainty_grid(
                    self.ensemble,
                    self.dataset.X_all,
                    self.dataset.n_lon,
                    self.dataset.n_lat,
                    iteration,
                    train_x=self.train_x,
                    save_path=(
                        self.figure_dir
                        / f"hyperparam_uncertainty_{self.acquisition}_iter{iteration}.png"
                    ),
                )
                plot_acquisition_comparison(
                    self.ensemble,
                    self.dataset.X_all,
                    self.dataset.n_lon,
                    self.dataset.n_lat,
                    iteration,
                    train_x=self.train_x,
                    save_path=(
                        self.figure_dir
                        / f"acquisition_comparison_{self.acquisition}_iter{iteration}.png"
                    ),
                    batch_size=config.batch_size,
                )

            best_idx, _ = self.pick_next_point(batch_size=config.batch_size)
            self.update_dataset(best_idx)

        return self.results
