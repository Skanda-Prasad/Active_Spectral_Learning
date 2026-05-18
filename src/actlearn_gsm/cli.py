"""Command-line interface for the active-learning experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from actlearn_gsm.config import (
    ActiveLearningConfig,
    BaselineConfig,
    EnsembleConfig,
    ExperimentConfig,
    GridConfig,
)
from actlearn_gsm.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MC-EV vs predictive-variance active learning.",
    )
    parser.add_argument("--n-lon", type=int, default=GridConfig.n_lon)
    parser.add_argument("--n-lat", type=int, default=GridConfig.n_lat)
    parser.add_argument("--grid-seed", type=int, default=GridConfig.seed)
    parser.add_argument("--ensemble-size", type=int, default=EnsembleConfig.size)
    parser.add_argument("--num-inducing", type=int, default=EnsembleConfig.num_inducing)
    parser.add_argument("--num-mixtures", type=int, default=EnsembleConfig.num_mixtures)
    parser.add_argument("--map-iters", type=int, default=EnsembleConfig.map_iters)
    parser.add_argument("--lr", type=float, default=EnsembleConfig.lr)
    parser.add_argument("--sm-iters", type=int, default=BaselineConfig.iters)
    parser.add_argument("--sm-lr", type=float, default=BaselineConfig.lr)
    parser.add_argument("--num-init", type=int, default=ActiveLearningConfig.num_init)
    parser.add_argument("--num-steps", type=int, default=ActiveLearningConfig.num_steps)
    parser.add_argument("--plot-every", type=int, default=ActiveLearningConfig.plot_every)
    parser.add_argument("--al-seed", type=int, default=ActiveLearningConfig.seed)
    parser.add_argument("--batch-size", type=int, default=ActiveLearningConfig.batch_size)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    grid = GridConfig(n_lon=args.n_lon, n_lat=args.n_lat, seed=args.grid_seed)
    ensemble = EnsembleConfig(
        size=args.ensemble_size,
        num_inducing=args.num_inducing,
        num_mixtures=args.num_mixtures,
        map_iters=args.map_iters,
        lr=args.lr,
    )
    baseline = BaselineConfig(
        num_mixtures=args.num_mixtures,
        iters=args.sm_iters,
        lr=args.sm_lr,
    )
    active_learning = ActiveLearningConfig(
        num_init=args.num_init,
        num_steps=args.num_steps,
        plot_every=args.plot_every,
        seed=args.al_seed,
        batch_size=args.batch_size,
    )
    return replace(
        ExperimentConfig(),
        grid=grid,
        ensemble=ensemble,
        baseline=baseline,
        active_learning=active_learning,
        artifact_dir=args.artifact_dir,
        make_plots=not args.no_plots,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_experiment(config_from_args(args))


if __name__ == "__main__":
    main()
