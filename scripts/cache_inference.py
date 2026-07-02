#!/usr/bin/env python3
"""Cache LieFlow inference outputs for fast strategy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lieflow_quant.cache import save_inference_cache
from lieflow_quant.inference import infer_cloud_symmetry, load_so3_model
from lieflow_quant.panel import load_cross_sections_from_npy


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache LieFlow inference for OOS days.")
    parser.add_argument("--config-dir", type=Path, default=Path("vendor/lieflow/conf"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("vendor/lieflow/outputs/2026-07-01/16-25-52/ckpt/model.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/equity"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/strategy/inference_cache.npz"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split",
        choices=("tail", "head", "all"),
        default="all",
        help="tail=OOS tail only (legacy), head=IS only, all=full sample (filter at eval via ML test split)",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--n-mc", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sections = load_cross_sections_from_npy(args.data_dir)
    if args.split == "all" or args.test_ratio <= 0:
        pass
    elif args.split == "tail" and 0 < args.test_ratio < 1:
        split = int(len(sections) * (1 - args.test_ratio))
        sections = sections[split:]
    elif args.split == "head" and 0 < args.test_ratio < 1:
        split = int(len(sections) * (1 - args.test_ratio))
        sections = sections[:split]

    if args.start_date or args.end_date:
        start = pd.Timestamp(args.start_date) if args.start_date else pd.Timestamp.min
        end = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.max
        sections = [s for s in sections if start <= pd.Timestamp(s.date).normalize() <= end]

    model, n_steps, _ = load_so3_model(
        args.config_dir,
        args.checkpoint,
        device=args.device,
        data_path=args.data_dir / "clouds.npy",
        metadata_path=args.data_dir / "metadata.npz",
    )

    rows: list[dict] = []
    for day_idx, section in enumerate(tqdm(sections, desc="Caching inference")):
        out = infer_cloud_symmetry(
            model,
            section.cloud,
            device=args.device,
            n_steps=n_steps,
            n_mc=args.n_mc,
            seed=args.seed + day_idx,
        )
        for ticker, mom, vol, radial in zip(
            section.tickers,
            out["canonical_momentum"],
            out["canonical_vol"],
            out["radial_distance"],
        ):
            rows.append(
                {
                    "date": section.date,
                    "ticker": ticker,
                    "canonical_momentum": float(mom),
                    "canonical_vol": float(vol),
                    "radial_distance": float(radial),
                    "concentration": out["concentration"],
                    "z_rotation_median_deg": out["z_rotation_median_deg"],
                    "regime": section.regime,
                    "vix": section.vix,
                }
            )

    save_inference_cache(rows, args.output)
    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
