"""GSM ensemble and acquisition utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gpytorch
import torch

from actlearn_gsm.config import EnsembleConfig
from actlearn_gsm.kernels import GSMFactorizedKernel2D
from actlearn_gsm.models import ExactGP2D, build_gsm_model


Logger = Callable[[str], None]


@dataclass
class EnsembleMember:
    model: ExactGP2D
    likelihood: gpytorch.likelihoods.GaussianLikelihood


class GSMEnsemble:
    """Multi-start MAP ensemble that approximates posterior variation."""

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self.members: list[EnsembleMember] = []

    def build(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        logger: Logger | None = print,
    ) -> None:
        if logger:
            logger("")
            logger("=" * 60)
            logger("BUILDING GSM ENSEMBLE (multi-start MAP)")
            logger("=" * 60)

        self.members = []
        for member_index in range(self.config.size):
            if logger:
                logger(
                    f"  >>> Training GSM member "
                    f"{member_index + 1}/{self.config.size}"
                )

            model, likelihood = build_gsm_model(train_x, train_y, self.config)
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for iteration in range(1, self.config.map_iters + 1):
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-2):
                    output = model(train_x)
                    loss = -mll(output, train_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                should_log = iteration % 50 == 0 or iteration == self.config.map_iters
                if logger and should_log:
                    logger(
                        f"    iter {iteration}/{self.config.map_iters}, "
                        f"loss={loss.item():.4f}"
                    )

            model.eval()
            likelihood.eval()
            self.members.append(EnsembleMember(model=model, likelihood=likelihood))

        if logger:
            logger("  Ensemble built.")

    @torch.no_grad()
    def ensemble_predict_means(
        self,
        X: torch.Tensor,
        *,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Return posterior means with shape (num_members, num_points)."""
        means = []
        for member in self.members:
            preds = []
            for start in range(0, X.shape[0], batch_size):
                xb = X[start : start + batch_size]
                with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
                    1e-2
                ):
                    output = member.likelihood(member.model(xb))
                preds.append(output.mean.detach())
            means.append(torch.cat(preds, dim=0))
        return torch.stack(means, dim=0)

    @torch.no_grad()
    def mc_epistemic_variance(
        self,
        X: torch.Tensor,
        *,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """MC-EV score: variance across ensemble posterior means."""
        means = self.ensemble_predict_means(X, batch_size=batch_size)
        return means.var(dim=0, unbiased=means.shape[0] > 1)

    @torch.no_grad()
    def predictive_variance_single_model(
        self,
        X: torch.Tensor,
        *,
        member_index: int = 0,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Predictive-variance acquisition for a fixed ensemble member."""
        member = self.members[member_index]
        variances = []
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size]
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
                1e-2
            ):
                output = member.likelihood(member.model(xb))
            variances.append(output.variance.detach())
        return torch.cat(variances, dim=0)

    @torch.no_grad()
    def parameter_disagreement_decomposed(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Hyperparameter variance by coordinate and parameter family."""
        lon = X[:, 0:1]
        lat = X[:, 1:2]
        values: dict[str, list[torch.Tensor]] = {
            "w_lon": [],
            "mu_lon": [],
            "ell_lon": [],
            "w_lat": [],
            "mu_lat": [],
            "ell_lat": [],
        }

        for member in self.members:
            kernel = member.model.covar_module
            values["w_lon"].append(
                GSMFactorizedKernel2D.transform_w(kernel.hyper_w_lon(lon))
            )
            values["mu_lon"].append(
                GSMFactorizedKernel2D.transform_mu(kernel.hyper_mu_lon(lon))
            )
            values["ell_lon"].append(
                GSMFactorizedKernel2D.transform_ell(kernel.hyper_ell_lon(lon))
            )
            values["w_lat"].append(
                GSMFactorizedKernel2D.transform_w(kernel.hyper_w_lat(lat))
            )
            values["mu_lat"].append(
                GSMFactorizedKernel2D.transform_mu(kernel.hyper_mu_lat(lat))
            )
            values["ell_lat"].append(
                GSMFactorizedKernel2D.transform_ell(kernel.hyper_ell_lat(lat))
            )

        result = {}
        for key, tensors in values.items():
            stacked = torch.stack(tensors, dim=0)
            result[key] = stacked.var(dim=0, unbiased=stacked.shape[0] > 1).mean(dim=1)
        return result
