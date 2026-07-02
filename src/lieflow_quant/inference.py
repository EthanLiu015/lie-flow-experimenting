"""LieFlow model loading and per-cloud inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

from lieflow.utils import seed_all


def _checkpoint_n_steps(checkpoint: Path, fallback: int) -> int:
    """Prefer ``n_steps`` saved with the checkpoint over current yaml defaults."""
    hydra_config = checkpoint.resolve().parent.parent / ".hydra" / "config.yaml"
    if not hydra_config.exists():
        return fallback
    try:
        from omegaconf import OmegaConf

        saved = OmegaConf.load(hydra_config)
        return int(saved.test.n_steps)
    except (AttributeError, TypeError, ValueError):
        return fallback


def load_so3_model(
    config_dir: Path | str,
    checkpoint: Path | str,
    device: str = "cpu",
    *,
    dataset: str = "SO3_equity_cross_section",
    model: str = "flow_matching/SO3_equity_cross_section",
    data_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> tuple[torch.nn.Module, int, object]:
    """Instantiate and load a trained SO(3) LieFlow checkpoint."""
    config_dir = Path(config_dir)
    overrides = [
        f"dataset={dataset}",
        f"model={model}",
        f"device={device}",
    ]
    if data_path is not None:
        overrides.append(f"dataset.train.data_path={data_path}")
        overrides.append(f"dataset.test.data_path={data_path}")
    if metadata_path is not None:
        overrides.append(f"dataset.train.metadata_path={metadata_path}")
        overrides.append(f"dataset.test.metadata_path={metadata_path}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name="train", overrides=overrides)
        seed_all(cfg.seed)
        net = instantiate(cfg.model).to(device)
        net.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        net.eval()
        n_steps = _checkpoint_n_steps(Path(checkpoint), int(cfg.test.n_steps))
        return net, n_steps, cfg


def z_rotation_angle_deg(rot_mat: np.ndarray) -> float:
    r = Rotation.from_matrix(rot_mat)
    return float(r.as_euler("zyx", degrees=True)[0] % 360)


def canonicalize_transform(
    orig_tf: torch.Tensor,
    total_tf: torch.Tensor,
) -> torch.Tensor:
    """Match 2D evaluation: (orig^T @ total)^T applied to clouds."""
    return (orig_tf.adjoint() @ total_tf).adjoint()


def infer_cloud_symmetry(
    model: torch.nn.Module,
    cloud: np.ndarray,
    *,
    device: str = "cpu",
    n_steps: int = 30,
    n_mc: int = 16,
    seed: int | None = None,
) -> dict:
    """
    Run LieFlow on one daily cross-section cloud.

    Returns canonical momentum residuals per stock, z-rotation angles from MC
    samples, and a concentration score for the exposure overlay.
    """
    if seed is not None:
        seed_all(seed)

    x = torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0).to(device)
    x_batch = x.repeat(n_mc, 1, 1)

    with torch.no_grad():
        _, orig_tf, total_tf = model.sample(
            x_batch, n_steps, return_transform=True
        )

    canon_tf = canonicalize_transform(orig_tf, total_tf)
    x_canon = torch.bmm(x_batch, canon_tf.transpose(-2, -1))
    x_canon_np = x_canon.cpu().numpy()

    # Use median MC draw for per-stock residuals (robust to sampling noise).
    canon_momentum = np.median(x_canon_np[:, :, 0], axis=0)
    canon_vol = np.median(x_canon_np[:, :, 1], axis=0)
    centroid = np.median(x_canon_np, axis=0).mean(axis=0)
    radial = np.linalg.norm(np.median(x_canon_np, axis=0) - centroid, axis=1)

    rot_mats = total_tf.cpu().numpy()
    z_angles = Rotation.from_matrix(rot_mats).as_euler("zyx", degrees=True)[:, 0] % 360
    hist, _ = np.histogram(z_angles, bins=36, range=(0, 360))
    concentration = float(hist.max() / max(hist.sum(), 1))

    return {
        "canonical_momentum": canon_momentum.astype(np.float32),
        "canonical_vol": canon_vol.astype(np.float32),
        "radial_distance": radial.astype(np.float32),
        "z_rotation_angles_deg": z_angles.astype(np.float32),
        "concentration": concentration,
        "z_rotation_median_deg": float(np.median(z_angles)),
    }
