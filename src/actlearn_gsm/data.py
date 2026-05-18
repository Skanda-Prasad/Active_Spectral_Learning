"""Synthetic geospatial data generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from actlearn_gsm.config import GridConfig


@dataclass(frozen=True)
class GeospatialDataset:
    X_all: torch.Tensor
    y_clean: torch.Tensor
    y_noisy: torch.Tensor
    lon_grid: torch.Tensor
    lat_grid: torch.Tensor
    n_lon: int
    n_lat: int

    @property
    def size(self) -> int:
        return int(self.X_all.shape[0])


def generate_geospatial_grid(
    config: GridConfig | None = None,
    *,
    device: torch.device | str | None = None,
) -> GeospatialDataset:
    """Generate a non-stationary synthetic 2D field on [0, 1]^2."""
    config = config or GridConfig()
    device = torch.device(device or "cpu")

    torch.manual_seed(config.seed)

    lon = torch.linspace(0.0, 1.0, config.n_lon, device=device)
    lat = torch.linspace(0.0, 1.0, config.n_lat, device=device)
    lon_mesh, lat_mesh = torch.meshgrid(lon, lat, indexing="ij")

    base_trend = 1.5 - 1.0 * lat_mesh
    zonal_band = (
        0.6
        * torch.sin(4 * math.pi * lon_mesh)
        * torch.exp(-((lat_mesh - 0.4) ** 2) / 0.02)
    )
    hotspot = (
        0.9
        * torch.sin(10 * math.pi * lon_mesh)
        * torch.exp(-(((lon_mesh - 0.8) ** 2) + ((lat_mesh - 0.2) ** 2)) / 0.01)
    )
    meridional = (
        0.8
        * torch.cos(6 * math.pi * lat_mesh)
        * (lon_mesh > 0.6).float()
        * (lat_mesh > 0.5).float()
    )

    field_clean = base_trend + zonal_band + hotspot + meridional
    noise_std = (
        0.05
        + 0.4 * lon_mesh
        + 0.3 * (lat_mesh**2)
        + 0.3 * torch.sin(3 * math.pi * lon_mesh * lat_mesh)
    ).clamp_min(1e-3)
    field_noisy = field_clean + noise_std * torch.randn(
        config.n_lon,
        config.n_lat,
        device=device,
    )

    return GeospatialDataset(
        X_all=torch.stack([lon_mesh.reshape(-1), lat_mesh.reshape(-1)], dim=-1),
        y_clean=field_clean.reshape(-1),
        y_noisy=field_noisy.reshape(-1),
        lon_grid=lon,
        lat_grid=lat,
        n_lon=config.n_lon,
        n_lat=config.n_lat,
    )
