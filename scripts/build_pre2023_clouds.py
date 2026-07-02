#!/usr/bin/env python3
"""Slice equity clouds to pre-ML-test dates for honest LieFlow training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lieflow_quant.methodology import ml_temporal_test_start
from lieflow_quant.panel import load_cross_sections_from_npy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/equity"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--cutoff-date",
        default=None,
        help="Inclusive last train date (default: day before temporal ML test start)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    meta = np.load(data_dir / "metadata.npz", allow_pickle=True)
    dates = pd.to_datetime(meta["dates"])
    clouds = np.load(data_dir / "clouds.npy")

    if args.cutoff_date:
        cutoff = pd.Timestamp(args.cutoff_date).normalize()
    else:
        cutoff = ml_temporal_test_start(dates, train_ratio=args.train_ratio) - pd.Timedelta(days=1)

    mask = dates <= cutoff
    n = int(mask.sum())
    if n < 100:
        raise SystemExit(f"Too few pre-cutoff days: {n}")

    out_clouds = clouds[mask]
    out_meta = {k: meta[k][mask] if meta[k].ndim > 0 and len(meta[k]) == len(dates) else meta[k] for k in meta.files}

    np.save(data_dir / "clouds_pre2023.npy", out_clouds)
    np.savez(data_dir / "metadata_pre2023.npz", **out_meta)
    print(f"cutoff={cutoff.date()} days={n} shape={out_clouds.shape}")
    print(f"Wrote {data_dir / 'clouds_pre2023.npy'}")


if __name__ == "__main__":
    main()
