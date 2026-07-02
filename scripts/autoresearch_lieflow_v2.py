#!/usr/bin/env python3
"""Autoresearch: LieFlow advanced strategies, optimize min Sharpe across all windows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from lieflow_quant.session import EvalSession, config_from_argv

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs" / "strategy" / "best_multiwindow_config.json"

# Each experiment: description + CLI args for eval_multiwindow.py
EXPERIMENTS: list[dict] = [
    {"description": "baseline radial smooth10", "args": ["--strategy", "radial_distance", "--signal-smoothing", "10"]},
    {"description": "radial smooth20", "args": ["--strategy", "radial_distance", "--signal-smoothing", "20"]},
    {"description": "radial sign flip", "args": ["--strategy", "radial_distance", "--signal-sign", "-1", "--signal-smoothing", "10"]},
    {"description": "mom_resid_vol smooth10", "args": ["--strategy", "mom_resid_vol", "--signal-smoothing", "10"]},
    {"description": "mom_resid_vol smooth20", "args": ["--strategy", "mom_resid_vol", "--signal-smoothing", "20"]},
    {"description": "mom_resid_vol flip", "args": ["--strategy", "mom_resid_vol", "--signal-sign", "-1", "--signal-smoothing", "10"]},
    {"description": "mom_minus_vol smooth10", "args": ["--strategy", "mom_minus_vol", "--signal-smoothing", "10"]},
    {"description": "delta_momentum smooth5", "args": ["--strategy", "delta_momentum", "--signal-smoothing", "5"]},
    {"description": "delta_radial smooth5", "args": ["--strategy", "delta_radial", "--signal-smoothing", "5"]},
    {"description": "delta_vol smooth5", "args": ["--strategy", "delta_vol", "--signal-smoothing", "5"]},
    {"description": "radial_gated conc>=1.0", "args": ["--strategy", "radial_gated", "--min-concentration-ratio", "1.0", "--signal-smoothing", "10"]},
    {"description": "radial_gated conc>=1.2", "args": ["--strategy", "radial_gated", "--min-concentration-ratio", "1.2", "--signal-smoothing", "10"]},
    {"description": "mom_resid_gated conc>=1.0", "args": ["--strategy", "mom_resid_gated", "--min-concentration-ratio", "1.0", "--signal-smoothing", "10"]},
    {"description": "conc_scaled_momentum smooth10", "args": ["--strategy", "conc_scaled_momentum", "--signal-smoothing", "10"]},
    {"description": "canonical_mom smooth20", "args": ["--strategy", "canonical_momentum", "--signal-smoothing", "20"]},
    {"description": "hybrid vol conc>=1.0", "args": ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "20"]},
    {"description": "hybrid vol conc>=1.2 w0.7", "args": ["--hybrid-vol", "--min-concentration-ratio", "1.2", "--lieflow-weight", "0.7", "--signal-smoothing", "20"]},
    {"description": "radial lag2 smooth10", "args": ["--strategy", "radial_distance", "--signal-smoothing", "10", "--lag", "2"]},
    {"description": "mom_resid cost5 smooth10", "args": ["--strategy", "mom_resid_vol", "--signal-smoothing", "10", "--cost-bps", "5"]},
    {"description": "radial regime mid", "args": ["--strategy", "radial_distance", "--regime-filter", "mid", "--signal-smoothing", "10"]},
    {"description": "delta_radial smooth10 flip", "args": ["--strategy", "delta_radial", "--signal-sign", "-1", "--signal-smoothing", "10"]},
    {"description": "mom_resid smooth5 cost10", "args": ["--strategy", "mom_resid_vol", "--signal-smoothing", "5"]},
    {"description": "radial smooth15 gated1.1", "args": ["--strategy", "radial_gated", "--min-concentration-ratio", "1.1", "--signal-smoothing", "15"]},
    {"description": "mom_minus_vol flip smooth15", "args": ["--strategy", "mom_minus_vol", "--signal-sign", "-1", "--signal-smoothing", "15"]},
    {"description": "delta_momentum smooth10 flip", "args": ["--strategy", "delta_momentum", "--signal-sign", "-1", "--signal-smoothing", "10"]},
]


def run_eval(session: EvalSession, extra: list[str]) -> dict:
    return session.evaluate_multiwindow(config_from_argv(extra))


def guard(m: dict) -> None:
    assert m["n_periods"] == 6, "expected 6 validation windows"
    for name, pm in m["periods"].items():
        assert pm.get("n_days", 0) >= 200, f"too few days in {name}"


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "-"


def main() -> None:
    print("[autoresearch] mode: classic")
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel sweep workers (default: 1 serial; use >1 for parallel batch)",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = args.output_dir or ROOT / "autoresearch" / f"loop-lieflow-v2-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    with open(tsv, "w") as f:
        f.write("# metric_direction: higher_is_better (min_sharpe across 6 windows @ 10bps)\n")
        f.write("iteration\ttimestamp\tcommit\tmin_sharpe\tmean_sharpe\tall_pos\tguard\tstatus\tdescription\n")

    best_min = float("-inf")
    best: dict | None = None
    best_args: list[str] = []
    kept = discarded = 0
    sha = git_head()
    converged = False

    print(f"Loading EvalSession (panel + cache once)...")
    session = EvalSession(
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        data_dir=ROOT / "data/equity",
        n_target=60,
    )
    print(f"Session ready in {session.load_ms:.0f}ms ({session.stats.n_cache_rows} rows)")

    batch = EXPERIMENTS[: args.iterations]
    workers = args.workers if args.workers is not None else 1

    if workers > 1:
        print(f"[autoresearch] parallel batch sweep ({workers} workers, {len(batch)} configs)")
        configs = [config_from_argv(exp["args"]) for exp in batch]
        t0 = perf_counter()
        metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
        sweep_ms = (perf_counter() - t0) * 1000
        print(f"Sweep finished in {sweep_ms:.0f}ms ({sweep_ms / len(batch):.0f}ms/config)")

        for i, (exp, m) in enumerate(zip(batch, metrics_list)):
            desc = exp["description"]
            extra = exp["args"]
            try:
                guard(m)
                min_s = float(m["min_sharpe"])
                mean_s = float(m["mean_sharpe"])
                all_pos = m["all_windows_positive"]

                if min_s >= best_min:
                    status = "keep"
                    kept += 1
                    best_min = min_s
                    best = m
                    best_args = extra
                else:
                    status = "discard"
                    discarded += 1

                with open(tsv, "a") as f:
                    f.write(
                        f"{i}\t{datetime.now().isoformat()}\t{sha}\t{min_s:.6f}\t{mean_s:.6f}\t"
                        f"{all_pos}\tpass\t{status}\t{desc}\n"
                    )

                print(
                    f"[iter {i}] {desc}: min_sharpe={min_s:.4f} mean={mean_s:.4f} "
                    f"all_pos={all_pos} ({status})"
                )

                if all_pos and min_s > 0:
                    converged = True
            except Exception as exc:
                with open(tsv, "a") as f:
                    f.write(f"{i}\t{datetime.now().isoformat()}\t-\t-\t-\t-\tfail\tcrash\t{desc}: {exc}\n")
                print(f"[iter {i}] {desc}: CRASH {exc}")

        if converged:
            print(f"CONVERGED: at least one config with all 6 windows positive (best min_sharpe={best_min:.4f})")
    else:
        for i, exp in enumerate(batch):
            desc = exp["description"]
            extra = exp["args"]
            try:
                m = run_eval(session, extra)
                guard(m)
                min_s = float(m["min_sharpe"])
                mean_s = float(m["mean_sharpe"])
                all_pos = m["all_windows_positive"]

                if min_s >= best_min:
                    status = "keep"
                    kept += 1
                    best_min = min_s
                    best = m
                    best_args = extra
                else:
                    status = "discard"
                    discarded += 1

                with open(tsv, "a") as f:
                    f.write(
                        f"{i}\t{datetime.now().isoformat()}\t{sha}\t{min_s:.6f}\t{mean_s:.6f}\t"
                        f"{all_pos}\tpass\t{status}\t{desc}\n"
                    )

                print(
                    f"[iter {i}] {desc}: min_sharpe={min_s:.4f} mean={mean_s:.4f} "
                    f"all_pos={all_pos} ({status})"
                )

                if all_pos and min_s > 0:
                    print(f"CONVERGED: all 6 windows positive, min_sharpe={min_s:.4f}")
                    converged = True
                    break
            except Exception as exc:
                with open(tsv, "a") as f:
                    f.write(f"{i}\t{datetime.now().isoformat()}\t-\t-\t-\t-\tfail\tcrash\t{desc}: {exc}\n")
                print(f"[iter {i}] {desc}: CRASH {exc}")

    if best is not None:
        BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        BEST_PATH.write_text(
            json.dumps(
                {"args": best_args, "metrics": best, "description": "Best multi-window LieFlow config"},
                indent=2,
            )
        )

    summary = out_dir / "evals-summary.md"
    summary.write_text(
        f"# LieFlow v2 Autoresearch\n\n"
        f"- Metric: **min Sharpe** across 6 windows @ 10bps (n=60 universe)\n"
        f"- Best min Sharpe: **{best_min:.4f}**\n"
        f"- All windows positive: **{best.get('all_windows_positive') if best else False}**\n"
        f"- Kept: {kept} / Discarded: {discarded}\n"
        f"- Best args: `{best_args}`\n"
        f"- Converged: {'yes' if converged else 'no'}\n"
    )

    handoff = {
        "version": "2.1.0",
        "source": "loop-lieflow-v2",
        "timestamp": datetime.now().isoformat(),
        "status": "CONVERGED" if converged else "BOUNDED",
        "results_tsv": str(tsv),
        "findings": [
            f"best_min_sharpe={best_min:.4f}",
            f"kept={kept} discarded={discarded}",
            f"all_windows_positive={best.get('all_windows_positive') if best else False}",
        ],
        "config": {
            "goal": "positive Sharpe all validation windows",
            "metric": "min_sharpe",
            "direction": "higher_is_better",
        },
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    print(json.dumps(handoff, indent=2))


if __name__ == "__main__":
    main()
