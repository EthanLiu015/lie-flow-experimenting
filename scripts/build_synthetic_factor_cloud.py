#!/usr/bin/env python3
"""Generate synthetic factor cross-section data with imposed C4 symmetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lieflow_quant.objects import build_factor_cross_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
    )
    parser.add_argument("--n-stocks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical = build_factor_cross_section(n_stocks=args.n_stocks, seed=args.seed)
    out_path = args.output_dir / "factor_cross_section_canonical.npy"
    np.save(out_path, canonical)
    print(f"Saved canonical cross-section ({canonical.shape}) to {out_path}")


if __name__ == "__main__":
    main()
