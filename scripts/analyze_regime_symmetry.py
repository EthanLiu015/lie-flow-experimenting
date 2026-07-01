#!/usr/bin/env python3
"""Compare learned SO(3) z-axis rotation histograms across VIX regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

from lieflow.utils import sample_random_batch, seed_all


def z_rotation_angles_deg(rot_mats: np.ndarray) -> np.ndarray:
    """Extract z-axis Euler angles (degrees) from SO(3) matrices."""
    angles = []
    for mat in rot_mats:
        r = Rotation.from_matrix(mat)
        z_angle = r.as_euler("zyx", degrees=True)[0]
        angles.append(z_angle % 360)
    return np.array(angles)


def extract_z_angles(model, dataset, device, n_samples, n_steps) -> np.ndarray:
    model.eval()
    batch = sample_random_batch(dataset, min(n_samples, len(dataset)), device)
    with torch.no_grad():
        _, _, transforms = model.sample(batch, n_steps, return_transform=True)
    return z_rotation_angles_deg(transforms.cpu().numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/regime_analysis"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regimes = {
        "low": "SO3_equity_low",
        "mid": "SO3_equity_mid",
        "high": "SO3_equity_high",
    }
    results = {}

    config_dir = str(args.config_dir.resolve())
    project_root = args.config_dir.resolve().parents[2]
    clouds = project_root / "data/equity/clouds.npy"
    metadata = project_root / "data/equity/metadata.npz"

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=[
                "dataset=SO3_equity_cross_section",
                "model=flow_matching/SO3_equity_cross_section",
                f"device={args.device}",
            ],
        )
        model = instantiate(cfg.model).to(args.device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        )
        n_steps = cfg.test.n_steps

        for regime, dataset_name in regimes.items():
            seed_all(cfg.seed)
            ds_cfg = compose(
                config_name="train",
                overrides=[
                    f"dataset={dataset_name}",
                    "model=flow_matching/SO3_equity_cross_section",
                    f"device={args.device}",
                    f"dataset.train.data_path={clouds}",
                    f"dataset.train.metadata_path={metadata}",
                    f"dataset.test.data_path={clouds}",
                    f"dataset.test.metadata_path={metadata}",
                ],
            )
            test_ds = instantiate(ds_cfg.dataset.test)
            angles = extract_z_angles(
                model, test_ds, args.device, args.n_samples, n_steps
            )
            hist, edges = np.histogram(angles, bins=36, range=(0, 360))
            results[regime] = {
                "mean_angle": float(np.mean(angles)),
                "std_angle": float(np.std(angles)),
                "concentration": float(hist.max() / max(hist.sum(), 1)),
                "hist": hist.tolist(),
                "bin_edges": edges.tolist(),
            }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, regime in zip(axes, regimes):
        centers = (np.array(results[regime]["bin_edges"][:-1]) + np.array(results[regime]["bin_edges"][1:])) / 2
        ax.bar(centers, results[regime]["hist"], width=10, color="steelblue", alpha=0.8)
        ax.set_title(f"VIX {regime} (conc={results[regime]['concentration']:.3f})")
        ax.set_xlabel("z-rotation (deg)")
        for deg in [0, 90, 180, 270]:
            ax.axvline(deg, color="green", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Count")
    fig.suptitle("Learned z-axis rotation support by VIX regime")
    fig.tight_layout()
    fig.savefig(args.output_dir / "regime_histograms.png", dpi=150)
    plt.close(fig)

    summary = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("hist", "bin_edges")}
        for k, v in results.items()
    }
    with open(args.output_dir / "regime_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
