"""Point-cloud objects for finance-inspired LieFlow experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FileObject:
    """Load a canonical point cloud from a .npy file (shape N x D)."""

    def __init__(self, path: str):
        data = np.load(path)
        if data.ndim != 2:
            raise ValueError(f"Expected shape (N, D), got {data.shape}")
        self.data = data.astype(np.float32)
        self.name = Path(path).stem


def build_factor_cross_section(
    n_stocks: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Build an asymmetric canonical cross-section in (momentum, volatility) space.

    The shape is intentionally non-rotationally symmetric so that C4 symmetry
    must be imposed via data generation, mirroring LieFlow's C4 arrow experiment.
    """
    rng = np.random.default_rng(seed)

    # Three momentum-vol clusters forming an L-shaped fan (asymmetric).
    cluster_centers = np.array(
        [
            [0.55, 0.15],
            [0.20, 0.55],
            [-0.35, 0.25],
        ],
        dtype=np.float32,
    )
    per_cluster = [8, 7, 5]
    assert sum(per_cluster) == n_stocks

    points = []
    for center, count in zip(cluster_centers, per_cluster):
        noise = rng.normal(scale=0.06, size=(count, 2)).astype(np.float32)
        points.append(center + noise)

    cloud = np.vstack(points).astype(np.float32)

    # Center and scale to unit variance for stable training.
    cloud -= cloud.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(cloud, axis=1).max()
    if scale > 0:
        cloud /= scale

    return cloud
