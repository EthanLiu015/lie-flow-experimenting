#!/usr/bin/env python3
"""Build daily 3D cross-section point clouds from equity panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lieflow_quant.panel import (
    assign_expanding_vix_regime,
    build_daily_cross_sections,
    load_equity_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/equity"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/equity"))
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--min-stocks", type=int, default=40)
    parser.add_argument("--n-target", type=int, default=50)
    args = parser.parse_args()

    close, vix = load_equity_panel(args.input_dir)
    sections = build_daily_cross_sections(
        close,
        momentum_window=args.momentum_window,
        vol_window=args.vol_window,
        min_stocks=args.min_stocks,
        n_target=args.n_target,
        vix=vix,
    )

    clouds_arr = np.stack([s.cloud for s in sections], axis=0)
    dates_arr = np.array([s.date.isoformat() for s in sections])
    vix_arr = np.array(
        [np.nan if s.vix is None else s.vix for s in sections], dtype=np.float32
    )
    tickers_arr = np.array([s.tickers for s in sections], dtype=object)

    regime = assign_expanding_vix_regime(vix_arr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "clouds.npy", clouds_arr)
    np.savez(
        args.output_dir / "metadata.npz",
        dates=dates_arr,
        vix=vix_arr,
        regime=regime,
        tickers=tickers_arr,
    )
    print(
        f"Built {clouds_arr.shape[0]} daily clouds "
        f"({clouds_arr.shape[1]} stocks x 3 features) -> {args.output_dir}"
    )
    print(
        f"Regime counts: low={(regime=='low').sum()}, "
        f"mid={(regime=='mid').sum()}, high={(regime=='high').sum()}"
    )


if __name__ == "__main__":
    main()
