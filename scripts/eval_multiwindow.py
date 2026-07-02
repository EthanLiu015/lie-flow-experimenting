#!/usr/bin/env python3
"""Evaluate a strategy across all validation windows; optimize min Sharpe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lieflow_quant.session import EvalSession, config_from_namespace


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-window strategy evaluation.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/strategy/inference_cache_full_n60.npz"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/equity"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/ar_multiwindow.json"),
    )
    parser.add_argument("--strategy", default="canonical_vol")
    parser.add_argument("--signal-sign", type=float, default=1.0)
    parser.add_argument("--signal-smoothing", type=int, default=20)
    parser.add_argument("--concentration-window", type=int, default=60)
    parser.add_argument("--min-exposure", type=float, default=0.25)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--min-concentration-ratio", type=float, default=None)
    parser.add_argument("--regime-filter", default=None)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--n-target", type=int, default=60)
    parser.add_argument("--hybrid-vol", action="store_true")
    parser.add_argument("--raw-vol-baseline", action="store_true")
    parser.add_argument("--lieflow-weight", type=float, default=0.5)
    parser.add_argument(
        "--inference-ml-test-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LieFlow inference only from the model temporal test split (default: on).",
    )
    args = parser.parse_args()

    if not args.cache.exists():
        raise FileNotFoundError(f"Cache not found: {args.cache}")

    session = EvalSession(
        data_dir=args.data_dir,
        cache_path=args.cache,
        n_target=args.n_target,
        inference_ml_test_only=args.inference_ml_test_only,
    )
    metrics = session.evaluate_multiwindow(config_from_namespace(args))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
