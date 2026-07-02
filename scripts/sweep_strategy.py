#!/usr/bin/env python3
"""Sweep strategy configs and print ranked Sharpe results."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval_strategy.py"

EXPERIMENTS = [
    ("baseline", []),
    ("sign_flip", ["--signal-sign", "-1"]),
    ("radial", ["--signal-feature", "radial_distance"]),
    ("radial_flip", ["--signal-feature", "radial_distance", "--signal-sign", "-1"]),
    ("vol", ["--signal-feature", "canonical_vol"]),
    ("vol_flip", ["--signal-feature", "canonical_vol", "--signal-sign", "-1"]),
    ("flip_conc", ["--signal-sign", "-1", "--combine-concentration"]),
    ("flip_smooth3", ["--signal-sign", "-1", "--signal-smoothing", "3"]),
    ("flip_smooth5", ["--signal-sign", "-1", "--signal-smoothing", "5"]),
    ("flip_mid", ["--signal-sign", "-1", "--regime-filter", "mid"]),
    ("flip_lag2", ["--signal-sign", "-1", "--lag", "2"]),
    ("flip_hold5", ["--signal-sign", "-1", "--hold-days", "5"]),
    ("flip_hold5_h5", ["--signal-sign", "-1", "--hold-days", "5", "--forward-horizon", "5"]),
    ("flip_cost2", ["--signal-sign", "-1", "--cost-bps", "2"]),
    ("flip_combo", ["--signal-sign", "-1", "--signal-smoothing", "5", "--hold-days", "5", "--forward-horizon", "5"]),
    ("radial_combo", ["--signal-feature", "radial_distance", "--signal-sign", "-1", "--signal-smoothing", "5", "--hold-days", "5", "--forward-horizon", "5"]),
    ("vol_combo", ["--signal-feature", "canonical_vol", "--signal-sign", "-1", "--signal-smoothing", "5", "--hold-days", "5", "--forward-horizon", "5"]),
    ("smooth3_hold3_h3", ["--signal-sign", "-1", "--signal-smoothing", "3", "--hold-days", "3", "--forward-horizon", "3"]),
    ("smooth10_hold10_h10", ["--signal-sign", "-1", "--signal-smoothing", "10", "--hold-days", "10", "--forward-horizon", "10"]),
    ("flip_minexp05", ["--signal-sign", "-1", "--min-exposure", "0.5"]),
]


def main() -> None:
    results = []
    out = Path("/tmp/sweep_metrics.json")
    for name, args in EXPERIMENTS:
        cmd = [sys.executable, str(EVAL), "--output-json", str(out), *args]
        subprocess.run(cmd, cwd=ROOT, check=True)
        metrics = json.loads(out.read_text())
        metrics["name"] = name
        metrics["args"] = args
        results.append(metrics)
        print(
            f"{name:20s} sharpe={metrics['sharpe']:7.4f} "
            f"ic={metrics['mean_ic']:7.4f} turnover={metrics['mean_turnover']:.3f} "
            f"ret={metrics['total_return']:7.4f}"
        )

    results.sort(key=lambda m: m["sharpe"], reverse=True)
    best = results[0]
    print("\nBEST:", best["name"], best["sharpe"])
    (ROOT / "outputs" / "strategy" / "sweep_results.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()
