"""Hyperparameter uncertainty summaries."""

from __future__ import annotations

import torch

from actlearn_gsm.ensemble import GSMEnsemble
from actlearn_gsm.models import ExactGP2D


def extract_psi_vector(model: ExactGP2D) -> torch.Tensor:
    """Flatten the GSM hyperfunction inducing values for one model."""
    kernel = model.covar_module
    params = [
        kernel.hyper_w_lon.inducing_vals.view(-1),
        kernel.hyper_mu_lon.inducing_vals.view(-1),
        kernel.hyper_ell_lon.inducing_vals.view(-1),
        kernel.hyper_w_lat.inducing_vals.view(-1),
        kernel.hyper_mu_lat.inducing_vals.view(-1),
        kernel.hyper_ell_lat.inducing_vals.view(-1),
    ]
    return torch.cat(params, dim=0).detach().cpu()


def hyper_param_uncertainty(ensemble: GSMEnsemble) -> tuple[float, float]:
    """Return trace and log determinant of the ensemble covariance over psi."""
    psi = torch.stack(
        [extract_psi_vector(member.model) for member in ensemble.members],
        dim=0,
    )
    psi_centered = psi - psi.mean(dim=0, keepdim=True)
    covariance = psi_centered.t() @ psi_centered / (psi.size(0) - 1 + 1e-8)

    trace = covariance.diag().sum().item()
    jitter = 1e-6 * torch.eye(covariance.size(0), dtype=covariance.dtype)
    _, logdet = torch.linalg.slogdet(covariance + jitter)
    return trace, logdet.item()
