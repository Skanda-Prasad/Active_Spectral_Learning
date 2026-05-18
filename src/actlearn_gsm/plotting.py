"""Plotting helpers for experiment diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from actlearn_gsm.ensemble import GSMEnsemble


def save_figure(fig: plt.Figure, save_path: Path | str | None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _finish_plot(fig: plt.Figure, save_path: Path | str | None, show: bool) -> None:
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_hyperparameter_uncertainty_grid(
    ensemble: GSMEnsemble,
    X_all: torch.Tensor,
    n_lon: int,
    n_lat: int,
    iteration: int,
    *,
    train_x: torch.Tensor | None = None,
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot heatmaps for GSM hyperparameter disagreement."""
    var_dict = ensemble.parameter_disagreement_decomposed(X_all)

    fig = plt.figure(figsize=(18, 10))
    grid = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    titles = [
        "Var[w(lon)]",
        "Var[mu(lon)]",
        "Var[ell(lon)]",
        "Var[w(lat)]",
        "Var[mu(lat)]",
        "Var[ell(lat)]",
    ]
    keys = ["w_lon", "mu_lon", "ell_lon", "w_lat", "mu_lat", "ell_lat"]

    train_np = train_x.detach().cpu().numpy() if train_x is not None else None
    for index, (key, title) in enumerate(zip(keys, titles, strict=True)):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        data = var_dict[key].detach().cpu().numpy().reshape(n_lon, n_lat).T
        image = ax.imshow(
            data,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap="plasma",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("lon_norm")
        ax.set_ylabel("lat_norm")
        if train_np is not None:
            ax.scatter(
                train_np[:, 0],
                train_np[:, 1],
                c="white",
                s=5,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.3,
            )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Hyperparameter Uncertainty Heatmaps (AL iter {iteration})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    _finish_plot(fig, save_path, show)


def plot_acquisition_comparison(
    ensemble: GSMEnsemble,
    X_all: torch.Tensor,
    n_lon: int,
    n_lat: int,
    iteration: int,
    *,
    train_x: torch.Tensor | None = None,
    save_path: Path | str | None = None,
    show: bool = False,
    batch_size: int = 512,
) -> None:
    """Compare MC-EV and predictive-variance acquisitions on the full grid."""
    param_acq = ensemble.mc_epistemic_variance(X_all, batch_size=batch_size)
    pred_acq = ensemble.predictive_variance_single_model(X_all, batch_size=batch_size)

    param_acq_norm = (param_acq - param_acq.min()) / (
        param_acq.max() - param_acq.min() + 1e-8
    )
    pred_acq_norm = (pred_acq - pred_acq.min()) / (
        pred_acq.max() - pred_acq.min() + 1e-8
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    train_np = train_x.detach().cpu().numpy() if train_x is not None else None

    param_grid = param_acq_norm.detach().cpu().numpy().reshape(n_lon, n_lat).T
    image = axes[0].imshow(
        param_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_title("MC-EV (Param Info Proxy)\nVar_psi[mu_psi(x)]")
    axes[0].set_xlabel("lon_norm")
    axes[0].set_ylabel("lat_norm")
    if train_np is not None:
        axes[0].scatter(
            train_np[:, 0],
            train_np[:, 1],
            c="red",
            s=10,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )
    plt.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    pred_grid = pred_acq_norm.detach().cpu().numpy().reshape(n_lon, n_lat).T
    image = axes[1].imshow(
        pred_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_title("Predictive Variance (Baseline)\nPred variance")
    axes[1].set_xlabel("lon_norm")
    axes[1].set_ylabel("lat_norm")
    if train_np is not None:
        axes[1].scatter(
            train_np[:, 0],
            train_np[:, 1],
            c="red",
            s=10,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )
    plt.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    image = axes[2].imshow(
        param_grid - pred_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    axes[2].set_title("Difference\nMC-EV - Pred Var")
    axes[2].set_xlabel("lon_norm")
    axes[2].set_ylabel("lat_norm")
    if train_np is not None:
        axes[2].scatter(
            train_np[:, 0],
            train_np[:, 1],
            c="yellow",
            s=10,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
    plt.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Acquisition Function Comparison (AL iter {iteration})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    _finish_plot(fig, save_path, show)


def plot_learning_curves_comparison(
    results_param: dict[str, list[float]],
    results_pred: dict[str, list[float]],
    *,
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot final comparison curves for both acquisition strategies."""
    fig = plt.figure(figsize=(16, 10))
    grid = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    iters_param = results_param["iter"]
    iters_pred = results_pred["iter"]

    ax0 = fig.add_subplot(grid[0, 0])
    ax0.plot(iters_param, results_param["mse_gsm"], "o-", label="MC-EV")
    ax0.plot(iters_pred, results_pred["mse_gsm"], "s--", label="Pred Var")
    ax0.set_xlabel("AL iteration")
    ax0.set_ylabel("GSM MSE")
    ax0.set_title("Predictive Performance", fontweight="bold")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.plot(iters_param, results_param["trace_cov"], "o-")
    ax1.plot(iters_pred, results_pred["trace_cov"], "s--")
    ax1.set_xlabel("AL iteration")
    ax1.set_ylabel("trace(Cov(psi))")
    ax1.set_title("Hyperparameter Uncertainty", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(grid[0, 2])
    ax2.plot(iters_param, results_param["logdet_cov"], "o-")
    ax2.plot(iters_pred, results_pred["logdet_cov"], "s--")
    ax2.set_xlabel("AL iteration")
    ax2.set_ylabel("logdet(Cov(psi))")
    ax2.set_title("Covariance Logdet", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(grid[1, 0])
    improvement_param = [
        sm / gsm for sm, gsm in zip(results_param["mse_sm"], results_param["mse_gsm"])
    ]
    improvement_pred = [
        sm / gsm for sm, gsm in zip(results_pred["mse_sm"], results_pred["mse_gsm"])
    ]
    ax3.plot(iters_param, improvement_param, "o-")
    ax3.plot(iters_pred, improvement_pred, "s--")
    ax3.axhline(y=1.0, color="red", linestyle=":", linewidth=2)
    ax3.set_xlabel("AL iteration")
    ax3.set_ylabel("SM MSE / GSM MSE")
    ax3.set_title("Advantage over Stationary SM", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.plot(iters_param, results_param["train_sizes"], "o-")
    ax4.set_xlabel("AL iteration")
    ax4.set_ylabel("Train set size")
    ax4.set_title("Data Acquired", fontweight="bold")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(grid[1, 2])
    mse_diff = [
        param - pred
        for param, pred in zip(results_param["mse_gsm"], results_pred["mse_gsm"])
    ]
    ax5.plot(iters_param, mse_diff, "o-")
    ax5.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax5.set_xlabel("AL iteration")
    ax5.set_ylabel("MSE(MC-EV) - MSE(Pred)")
    ax5.set_title("MSE Difference", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(grid[2, 0])
    ax6.plot(iters_param[1:], np.diff(results_param["trace_cov"]), "o-")
    ax6.plot(iters_pred[1:], np.diff(results_pred["trace_cov"]), "s--")
    ax6.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax6.set_xlabel("AL iteration")
    ax6.set_ylabel("delta trace(Cov(psi))")
    ax6.set_title("Trace Reduction Rate", fontweight="bold")
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(grid[2, 1])
    ax7.plot(iters_param, np.cumsum(results_param["mse_gsm"]), "o-")
    ax7.plot(iters_pred, np.cumsum(results_pred["mse_gsm"]), "s--")
    ax7.set_xlabel("AL iteration")
    ax7.set_ylabel("Cumulative MSE")
    ax7.set_title("Cumulative Prediction Error", fontweight="bold")
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(grid[2, 2])
    ax8.axis("off")
    final_mse_param = results_param["mse_gsm"][-1]
    final_mse_pred = results_pred["mse_gsm"][-1]
    final_trace_param = results_param["trace_cov"][-1]
    final_trace_pred = results_pred["trace_cov"][-1]
    table_data = [
        ["Metric", "MC-EV", "Pred Var", "Winner"],
        [
            "Final MSE",
            f"{final_mse_param:.4f}",
            f"{final_mse_pred:.4f}",
            "MC-EV" if final_mse_param < final_mse_pred else "Pred",
        ],
        [
            "Final trace",
            f"{final_trace_param:.4f}",
            f"{final_trace_pred:.4f}",
            "MC-EV" if final_trace_param < final_trace_pred else "Pred",
        ],
        [
            "Avg MSE",
            f"{np.mean(results_param['mse_gsm']):.4f}",
            f"{np.mean(results_pred['mse_gsm']):.4f}",
            (
                "MC-EV"
                if np.mean(results_param["mse_gsm"])
                < np.mean(results_pred["mse_gsm"])
                else "Pred"
            ),
        ],
    ]
    table = ax8.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for column in range(4):
        table[(0, column)].set_facecolor("#E0E0E0")
        table[(0, column)].set_text_props(weight="bold")
    ax8.set_title("Summary Statistics", fontweight="bold", pad=20)

    fig.suptitle(
        "Parameter-based MC-EV vs Prediction-based Active Learning",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    _finish_plot(fig, save_path, show)
