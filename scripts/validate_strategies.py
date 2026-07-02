#!/usr/bin/env python3
"""Walk-forward / out-of-regime validation for LieFlow vs raw-vol strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lieflow_quant.cache import load_inference_cache
from lieflow_quant.panel import load_equity_panel
from lieflow_quant.validation import (
    DEFAULT_COSTS_BPS,
    DEFAULT_PERIODS,
    build_lieflow_signals,
    build_panel_sections,
    build_raw_vol_signals_for_sections,
    run_validation_grid,
    save_validation_outputs,
    summarize_validation,
)

UNIVERSE_CONFIG: dict[int, tuple[Path, Path]] = {
    50: (
        Path("data/equity"),
        Path("outputs/strategy/inference_cache_full_n50.npz"),
    ),
    60: (
        Path("data/equity_univ60"),
        Path("outputs/strategy/inference_cache_full_n60.npz"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LieFlow vs raw-vol strategies.")
    parser.add_argument(
        "--universes",
        default="50,60",
        help="Comma-separated n_target values (e.g. 50,60)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/strategy/validation"))
    parser.add_argument("--signal-smoothing", type=int, default=20)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--costs-bps", default="5,10,15,20")
    args = parser.parse_args()

    costs = tuple(float(x) for x in args.costs_bps.split(","))
    universes = [int(x) for x in args.universes.split(",")]

    all_results = []
    close, vix = load_equity_panel("data/equity")

    for n_target in universes:
        if n_target not in UNIVERSE_CONFIG:
            raise ValueError(f"Unknown universe n_target={n_target}; supported: {list(UNIVERSE_CONFIG)}")
        data_dir, cache_path = UNIVERSE_CONFIG[n_target]
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cache for n={n_target}: {cache_path}\n"
                f"Run: python scripts/cache_inference.py --split all "
                f"--data-dir {data_dir} --output {cache_path} --device mps"
            )

        sections = build_panel_sections(close, n_target=n_target, vix=vix)
        section_dates = {pd.Timestamp(s.date).normalize() for s in sections}

        cache = load_inference_cache(cache_path)
        lieflow = build_lieflow_signals(cache, signal_smoothing=args.signal_smoothing)
        lieflow = lieflow[lieflow["date"].isin(section_dates)].copy()

        raw_vol = build_raw_vol_signals_for_sections(
            close,
            sections,
            signal_smoothing=args.signal_smoothing,
        )

        grid = run_validation_grid(
            lieflow_signals=lieflow,
            raw_vol_signals=raw_vol,
            close=close,
            periods=DEFAULT_PERIODS,
            costs_bps=costs or DEFAULT_COSTS_BPS,
            universe_label=f"n{n_target}",
            lag=args.lag,
        )
        all_results.append(grid)
        print(f"n={n_target}: {len(grid)} validation rows", flush=True)

    df = pd.concat(all_results, ignore_index=True)
    summary = summarize_validation(df)
    save_validation_outputs(df, summary, args.output_dir)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {args.output_dir}/validation_grid.csv")


if __name__ == "__main__":
    main()
