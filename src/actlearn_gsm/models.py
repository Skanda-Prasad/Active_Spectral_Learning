"""GP model builders and training helpers."""

from __future__ import annotations

from collections.abc import Callable

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal

from actlearn_gsm.config import BaselineConfig, EnsembleConfig
from actlearn_gsm.kernels import GSMFactorizedKernel2D, LearnableHyperfunction1D


Logger = Callable[[str], None]


class ExactGP2D(gpytorch.models.ExactGP):
    """Exact GP model with a constant mean and caller-provided kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def build_gsm_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    config: EnsembleConfig,
) -> tuple[ExactGP2D, gpytorch.likelihoods.GaussianLikelihood]:
    """Build one GSM ensemble member."""
    device = train_x.device
    dtype = train_x.dtype
    ind_lon = torch.linspace(0.0, 1.0, config.num_inducing, device=device, dtype=dtype)
    ind_lat = torch.linspace(0.0, 1.0, config.num_inducing, device=device, dtype=dtype)
    ind_lon = ind_lon.view(-1, 1)
    ind_lat = ind_lat.view(-1, 1)

    hyper_w_lon = LearnableHyperfunction1D(
        ind_lon,
        output_dim=config.num_mixtures,
    ).to(device)
    hyper_mu_lon = LearnableHyperfunction1D(
        ind_lon,
        output_dim=config.num_mixtures,
    ).to(device)
    hyper_ell_lon = LearnableHyperfunction1D(
        ind_lon,
        output_dim=config.num_mixtures,
    ).to(device)
    hyper_w_lat = LearnableHyperfunction1D(
        ind_lat,
        output_dim=config.num_mixtures,
    ).to(device)
    hyper_mu_lat = LearnableHyperfunction1D(
        ind_lat,
        output_dim=config.num_mixtures,
    ).to(device)
    hyper_ell_lat = LearnableHyperfunction1D(
        ind_lat,
        output_dim=config.num_mixtures,
    ).to(device)

    kernel = GSMFactorizedKernel2D(
        hyper_w_lon,
        hyper_mu_lon,
        hyper_ell_lon,
        hyper_w_lat,
        hyper_mu_lat,
        hyper_ell_lat,
        num_mixtures=config.num_mixtures,
    ).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGP2D(train_x, train_y, likelihood, kernel).to(device)
    return model, likelihood


def train_spectral_mixture_baseline(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    config: BaselineConfig,
    *,
    logger: Logger | None = print,
) -> tuple[ExactGP2D, gpytorch.likelihoods.GaussianLikelihood]:
    """Train the stationary spectral-mixture baseline."""
    if logger:
        logger("  Training SM baseline...")

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(train_x.device)
    kernel = gpytorch.kernels.SpectralMixtureKernel(
        num_mixtures=config.num_mixtures,
        ard_num_dims=2,
    ).to(train_x.device)
    kernel.initialize_from_data(train_x, train_y)
    model = ExactGP2D(train_x, train_y, likelihood, kernel).to(train_x.device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for iteration in range(1, config.iters + 1):
        optimizer.zero_grad()
        with gpytorch.settings.cholesky_jitter(1e-3):
            output = model(train_x)
            loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if logger and (iteration % 50 == 0 or iteration == config.iters):
            logger(
                f"    SM iter {iteration}/{config.iters}, loss={loss.item():.4f}"
            )

    model.eval()
    likelihood.eval()
    return model, likelihood
