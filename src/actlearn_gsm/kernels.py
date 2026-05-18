"""Custom kernels used by the active-learning experiment."""

from __future__ import annotations

import math

import gpytorch
import torch
from torch import nn


class LearnableHyperfunction1D(nn.Module):
    """Sparse interpolation model for a vector-valued 1D hyperfunction."""

    def __init__(
        self,
        inducing_x: torch.Tensor,
        init_scale: float = 0.5,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("inducing_x", inducing_x.clone())
        self.inducing_vals = nn.Parameter(
            torch.randn(
                inducing_x.shape[0],
                output_dim,
                device=inducing_x.device,
                dtype=inducing_x.dtype,
            )
            * init_scale
        )
        self.log_lengthscale = nn.Parameter(
            torch.log(
                torch.tensor(0.3, device=inducing_x.device, dtype=inducing_x.dtype)
            )
        )

    def rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        lengthscale = torch.exp(self.log_lengthscale)
        x1_scaled = x1 / lengthscale
        x2_scaled = x2.t() / lengthscale
        sqdist = (
            x1_scaled.pow(2).sum(-1, keepdim=True)
            + x2_scaled.pow(2).sum(0, keepdim=True)
            - 2 * x1_scaled @ x2_scaled
        )
        return torch.exp(-0.5 * sqdist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        k_uu = self.rbf_kernel(self.inducing_x, self.inducing_x)
        k_uu = k_uu + 1e-4 * torch.eye(
            k_uu.size(0),
            device=k_uu.device,
            dtype=k_uu.dtype,
        )
        k_xu = self.rbf_kernel(x, self.inducing_x)
        chol_uu = torch.linalg.cholesky(k_uu)
        alpha = torch.cholesky_solve(self.inducing_vals, chol_uu)
        return k_xu @ alpha


class GSMFactorizedKernel2D(gpytorch.kernels.Kernel):
    """Non-stationary factorized Gaussian spectral mixture kernel."""

    is_stationary = False

    def __init__(
        self,
        hyper_w_lon: LearnableHyperfunction1D,
        hyper_mu_lon: LearnableHyperfunction1D,
        hyper_ell_lon: LearnableHyperfunction1D,
        hyper_w_lat: LearnableHyperfunction1D,
        hyper_mu_lat: LearnableHyperfunction1D,
        hyper_ell_lat: LearnableHyperfunction1D,
        num_mixtures: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hyper_w_lon = hyper_w_lon
        self.hyper_mu_lon = hyper_mu_lon
        self.hyper_ell_lon = hyper_ell_lon
        self.hyper_w_lat = hyper_w_lat
        self.hyper_mu_lat = hyper_mu_lat
        self.hyper_ell_lat = hyper_ell_lat
        self.num_mixtures = num_mixtures

    @staticmethod
    def transform_w(raw: torch.Tensor) -> torch.Tensor:
        return 0.3 + 1.7 * torch.sigmoid(raw)

    @staticmethod
    def transform_mu(raw: torch.Tensor) -> torch.Tensor:
        return 0.2 + 2.8 * torch.sigmoid(raw)

    @staticmethod
    def transform_ell(raw: torch.Tensor) -> torch.Tensor:
        return 0.05 + 0.75 * torch.sigmoid(raw)

    def _gsm_1d(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        hyper_w: LearnableHyperfunction1D,
        hyper_mu: LearnableHyperfunction1D,
        hyper_ell: LearnableHyperfunction1D,
    ) -> torch.Tensor:
        n_rows = x1.shape[0]
        n_cols = x2.shape[0]

        w1 = self.transform_w(hyper_w(x1))
        w2 = self.transform_w(hyper_w(x2))
        mu1 = self.transform_mu(hyper_mu(x1))
        mu2 = self.transform_mu(hyper_mu(x2))
        ell1 = self.transform_ell(hyper_ell(x1))
        ell2 = self.transform_ell(hyper_ell(x2))

        dx = x1.expand(n_rows, n_cols) - x2.t().expand(n_rows, n_cols)
        dx = dx.unsqueeze(-1)

        w_prod = w1.unsqueeze(1) * w2.unsqueeze(0)
        mu_bar = 0.5 * (mu1.unsqueeze(1) + mu2.unsqueeze(0))
        ell_bar = 0.5 * (ell1.unsqueeze(1) + ell2.unsqueeze(0))

        exp_term = torch.exp(-2 * math.pi**2 * dx**2 / (ell_bar**2 + 1e-6))
        cos_term = torch.cos(2 * math.pi * mu_bar * dx)
        return (w_prod * exp_term * cos_term).sum(dim=2)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        x1 = x1.view(-1, 2)
        x2 = x2.view(-1, 2)
        lon1, lat1 = x1[:, 0:1], x1[:, 1:2]
        lon2, lat2 = x2[:, 0:1], x2[:, 1:2]

        k_lon = self._gsm_1d(
            lon1,
            lon2,
            self.hyper_w_lon,
            self.hyper_mu_lon,
            self.hyper_ell_lon,
        )
        k_lat = self._gsm_1d(
            lat1,
            lat2,
            self.hyper_w_lat,
            self.hyper_mu_lat,
            self.hyper_ell_lat,
        )
        kernel = k_lon * k_lat

        if diag:
            return kernel.diag()

        if kernel.shape[0] == kernel.shape[1]:
            kernel = kernel + 1e-3 * torch.eye(
                kernel.shape[0],
                device=kernel.device,
                dtype=kernel.dtype,
            )
        return kernel
