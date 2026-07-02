#!/usr/bin/env python3
"""Evaluate learned rotation angles from a LieFlow 2D checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from lieflow.geometry_2d import matrix_to_angle, polar_decomposition
from lieflow.utils import sample_random_batch, seed_all
from lieflow_quant.evaluation import angle_histogram_peaks, wasserstein_to_c4_angles


def extract_angles(model, test_dataset, device, n_samples: int, n_steps: int) -> np.ndarray:
    model.eval()
    batch, test_tf = sample_random_batch(
        test_dataset, n_samples, device, return_transform=True
    )
    with torch.no_grad():
        _, orig_tf, transforms = model.sample(
            batch, n_steps, return_transform=True
        )

    canonicalized = (orig_tf.adjoint() @ transforms).adjoint()
    r, _ = polar_decomposition(canonicalized)
    angles = np.rad2deg(matrix_to_angle(r).cpu().numpy().flatten())
    return angles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="C4_factor_cross_section")
    parser.add_argument("--model", default="flow_matching/SO2_to_C4_factor")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Flow integration steps (default: from checkpoint config)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    project_root = args.config_dir.resolve().parents[2]
    canonical = project_root / "data/synthetic/factor_cross_section_canonical.npy"

    with initialize_config_dir(
        version_base=None,
        config_dir=str(args.config_dir.resolve()),
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                f"dataset={args.dataset}",
                f"model={args.model}",
                f"device={args.device}",
                f"dataset.test.dist.base_object.path={canonical}",
            ],
        )

        seed_all(cfg.seed)
        test_dataset = instantiate(cfg.dataset.test)
        model = instantiate(cfg.model).to(args.device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        )

        n_steps = args.n_steps if args.n_steps is not None else int(cfg.test.n_steps)
        angles = extract_angles(
            model, test_dataset, args.device, args.n_samples, n_steps
        )
        metrics_n_steps = n_steps
    metrics = angle_histogram_peaks(angles)
    metrics["wasserstein_to_c4"] = wasserstein_to_c4_angles(angles)
    metrics["n_samples"] = args.n_samples
    metrics["n_steps"] = metrics_n_steps

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        metrics["bin_centers"],
        metrics["hist"],
        width=360 / len(metrics["hist"]),
        color="steelblue",
        alpha=0.8,
    )
    for peak in metrics["peak_angles_deg"]:
        ax.axvline(peak, color="crimson", linestyle="--", alpha=0.7)
    for target in [0, 90, 180, 270]:
        ax.axvline(target, color="green", linestyle=":", alpha=0.5)
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"C4 recall={metrics['c4_recall']:.2f}, "
        f"W1={metrics['wasserstein_to_c4']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "angles_histogram.png", dpi=150)
    plt.close(fig)

    summary = {k: v for k, v in metrics.items() if k not in ("hist", "bin_centers")}
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
