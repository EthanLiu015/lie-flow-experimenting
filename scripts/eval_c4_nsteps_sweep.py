#!/usr/bin/env python3
"""Sweep n_steps for C4 symmetry recovery eval at high sample count."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-steps-list", default="20,30,40,50")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/eval_synthetic_full")
    args = parser.parse_args()

    steps = [int(x) for x in args.n_steps_list.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    rows: list[dict] = []
    for n_steps in steps:
        out = args.output_dir / f"nsteps_{n_steps}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/evaluate_symmetry_recovery.py"),
            "--config-dir",
            str(ROOT / "vendor/lieflow/conf"),
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(out),
            "--device",
            args.device,
            "--n-samples",
            str(args.n_samples),
            "--n-steps",
            str(n_steps),
        ]
        print(f"\n>>> n_steps={n_steps} n_samples={args.n_samples}")
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)
        metrics = json.loads((out / "metrics.json").read_text())
        rows.append(
            {
                "n_steps": n_steps,
                "n_samples": args.n_samples,
                "c4_recall": metrics["c4_recall"],
                "n_peaks": metrics["n_peaks"],
                "peak_angles_deg": metrics["peak_angles_deg"],
                "concentration": metrics["concentration"],
                "wasserstein_to_c4": metrics["wasserstein_to_c4"],
            }
        )

    best = max(rows, key=lambda r: (r["c4_recall"], r["concentration"]))
    summary = {"runs": rows, "best": best}
    (args.output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SWEEP SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
