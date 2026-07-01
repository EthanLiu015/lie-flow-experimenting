#!/usr/bin/env python3
"""Generate daily LieFlow canonical-residual trading signals."""

from __future__ import annotations

import argparse
from pathlib import Path

from lieflow_quant.signals import run_signal_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-sectional trading signals from LieFlow symmetry inference."
    )
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
        default=Path("outputs/strategy/daily_signals.csv"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of trailing days to use (out-of-sample). Set 0 to use all days.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Optional cap on number of days (after test-ratio filter).",
    )
    parser.add_argument("--n-mc", type=int, default=16, help="MC samples per day for concentration.")
    parser.add_argument("--concentration-window", type=int, default=60)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train SO3_equity_cross_section first or pass --checkpoint."
        )

    signals = run_signal_pipeline(
        config_dir=args.config_dir,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        device=args.device,
        test_ratio=args.test_ratio,
        max_days=args.max_days,
        n_mc=args.n_mc,
        concentration_window=args.concentration_window,
    )
    print(f"Wrote {len(signals)} signal rows -> {args.output}")
    print(f"Date range: {signals['date'].min()} .. {signals['date'].max()}")
    print(f"Unique days: {signals['date'].nunique()}")


if __name__ == "__main__":
    main()
