#!/usr/bin/env python3
"""Round-2 autoresearch: fine-tune best LieFlow v2 candidates (warm EvalSession)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from lieflow_quant.session import EvalSession, config_from_argv

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("raw_vol baseline smooth20", ["--raw-vol-baseline", "--signal-smoothing", "20"]),
    ("radial smooth20", ["--strategy", "radial_distance", "--signal-smoothing", "20"]),
    ("radial smooth25", ["--strategy", "radial_distance", "--signal-smoothing", "25"]),
    ("canonical_mom smooth15", ["--strategy", "canonical_momentum", "--signal-smoothing", "15"]),
    ("canonical_mom smooth20", ["--strategy", "canonical_momentum", "--signal-smoothing", "20"]),
    ("mom_minus_vol flip s10", ["--strategy", "mom_minus_vol", "--signal-sign", "-1", "--signal-smoothing", "10"]),
    ("mom_minus_vol flip s15", ["--strategy", "mom_minus_vol", "--signal-sign", "-1", "--signal-smoothing", "15"]),
    ("mom_minus_vol flip s20", ["--strategy", "mom_minus_vol", "--signal-sign", "-1", "--signal-smoothing", "20"]),
    ("mom_minus_vol flip s25", ["--strategy", "mom_minus_vol", "--signal-sign", "-1", "--signal-smoothing", "25"]),
    ("mom_minus_vol flip cost5 s15", ["--strategy", "mom_minus_vol", "--signal-sign", "-1", "--signal-smoothing", "15", "--cost-bps", "5"]),
    ("radial smooth20 cost5", ["--strategy", "radial_distance", "--signal-smoothing", "20", "--cost-bps", "5"]),
    ("hybrid vol conc1.0 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "20"]),
    ("hybrid vol conc1.0 cost5", ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "20", "--cost-bps", "5"]),
    ("hybrid vol conc1.2 w0.5", ["--hybrid-vol", "--min-concentration-ratio", "1.2", "--lieflow-weight", "0.5", "--signal-smoothing", "20"]),
    ("radial_gated 1.05 s20", ["--strategy", "radial_gated", "--min-concentration-ratio", "1.05", "--signal-smoothing", "20"]),
    ("conc_scaled_mom s15", ["--strategy", "conc_scaled_momentum", "--signal-smoothing", "15"]),
    ("mom_resid s15 cost5", ["--strategy", "mom_resid_vol", "--signal-smoothing", "15", "--cost-bps", "5"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel sweep workers (default: cpu count)",
    )
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print(f"[autoresearch] mode: parallel sweep ({workers} workers)")
    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        n_target=60,
    )
    print(f"Session ready in {session.load_ms:.0f}ms")

    names = [name for name, _ in EXPERIMENTS]
    configs = [config_from_argv(argv) for _, argv in EXPERIMENTS]

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    sweep_ms = (perf_counter() - t0) * 1000
    print(f"Sweep finished in {sweep_ms:.0f}ms ({sweep_ms / len(configs):.0f}ms/config)")

    best_min = float("-inf")
    best_name = ""
    best_m: dict | None = None
    best_args: list[str] = []

    for name, argv, m in zip(names, [argv for _, argv in EXPERIMENTS], metrics_list):
        min_s = float(m["min_sharpe"])
        flag = "keep" if min_s >= best_min else "discard"
        if min_s >= best_min:
            best_min = min_s
            best_name = name
            best_m = m
            best_args = argv
        print(
            f"{name:35s} min={min_s:7.3f} mean={m['mean_sharpe']:7.3f} "
            f"all+={m['all_windows_positive']} pos={m['n_periods_positive']}/6 ({flag})"
        )

    print(f"\nBEST: {best_name} min_sharpe={best_min:.4f}")
    if best_m:
        path = ROOT / "outputs/strategy/best_multiwindow_config.json"
        path.write_text(
            json.dumps(
                {"name": best_name, "args": best_args, "metrics": best_m},
                indent=2,
            )
        )
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
