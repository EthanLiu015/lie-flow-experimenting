#!/usr/bin/env python3
"""
Benchmark cold-start vs warm-session latency for quant SWE demos.

Shows amortized I/O: load panel+cache once, then evaluate many configs in-process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lieflow_quant.session import EvalSession, MultiWindowConfig  # noqa: E402

SWEEP_CONFIGS = [
    MultiWindowConfig(strategy="radial_distance", signal_smoothing=s)
    for s in (10, 15, 20, 25)
] + [
    MultiWindowConfig(strategy="mom_minus_vol", signal_sign=-1, signal_smoothing=s)
    for s in (10, 15, 20, 25)
] + [
    MultiWindowConfig(strategy="mom_resid_vol", signal_smoothing=s)
    for s in (10, 15, 20)
] + [
    MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=20),
    MultiWindowConfig(hybrid_vol=True, min_concentration_ratio=1.0, signal_smoothing=20),
    MultiWindowConfig(hybrid_vol=True, min_concentration_ratio=1.2, lieflow_weight=0.5, signal_smoothing=20),
]


def _cold_subprocess_eval() -> float:
    t0 = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_multiwindow.py"),
            "--raw-vol-baseline",
            "--signal-smoothing",
            "20",
            "--output-json",
            "/tmp/bench_cold.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant SWE latency benchmark.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/equity"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/strategy/inference_cache_full_n60.npz"),
    )
    parser.add_argument("--output-json", type=Path, default=Path("outputs/benchmark/recruiting_metrics.json"))
    parser.add_argument("--parallel-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.cache.exists():
        raise FileNotFoundError(f"Cache not found: {args.cache}")

    # 1) Cold start: subprocess spawns fresh Python + reloads everything
    cold_ms = _cold_subprocess_eval()

    # 2) Warm session: load once
    t0 = time.perf_counter()
    session = EvalSession(data_dir=args.data_dir, cache_path=args.cache)
    session_load_ms = session.load_ms
    session_init_total_ms = (time.perf_counter() - t0) * 1000

    # 3) Warm single eval (amortized — no reload)
    warm = session.evaluate_multiwindow(
        MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=20),
        with_timing=True,
    )
    warm_eval_ms = warm["_timing"]["total_ms"]

    # 4) In-process sweep (autoresearch pattern)
    n_configs = len(SWEEP_CONFIGS)
    t0 = time.perf_counter()
    sweep_results = session.sweep_multiwindow(SWEEP_CONFIGS)
    sweep_total_ms = (time.perf_counter() - t0) * 1000
    sweep_per_config_ms = sweep_total_ms / n_configs

    # 5) Parallel sweep (worker pool loads session once per process)
    parallel_configs = [
        MultiWindowConfig(strategy="radial_distance", signal_smoothing=s)
        for s in (10, 20)
    ] + [
        MultiWindowConfig(strategy="mom_minus_vol", signal_sign=-1, signal_smoothing=s)
        for s in (15, 25)
    ] + [
        MultiWindowConfig(strategy="mom_resid_vol", signal_smoothing=10),
        MultiWindowConfig(strategy="canonical_vol", signal_smoothing=20),
    ]
    t0 = time.perf_counter()
    parallel_metrics = session.sweep_multiwindow(
        parallel_configs,
        with_timing=True,
        n_workers=args.parallel_workers,
    )
    parallel_total_ms = (time.perf_counter() - t0) * 1000
    parallel_results = [
        {
            "strategy": m["strategy"],
            "smooth": m["signal_smoothing"],
            "min_sharpe": m["min_sharpe"],
            "ms": m["_timing"]["total_ms"],
        }
        for m in parallel_metrics
    ]

    # Reference metric unchanged
    ref_min_sharpe = warm["min_sharpe"]

    report = {
        "historical_baseline": {
            "pre_optimization_cold_eval_ms": 24899,
            "note": "From first benchmark_latency.py run (6 separate backtests + subprocess overhead)",
        },
        "summary": {
            "cold_subprocess_eval_ms": round(cold_ms, 1),
            "warm_session_load_ms": round(session_load_ms, 1),
            "warm_single_eval_ms": round(warm_eval_ms, 1),
            "speedup_cold_vs_warm_eval": round(cold_ms / warm_eval_ms, 2),
            "sweep_n_configs": n_configs,
            "sweep_total_ms": round(sweep_total_ms, 1),
            "sweep_ms_per_config": round(sweep_per_config_ms, 1),
            "sweep_configs_per_sec": round(1000 / sweep_per_config_ms, 2),
            "parallel_n_jobs": len(parallel_configs),
            "parallel_total_ms": round(parallel_total_ms, 1),
            "parallel_ms_per_job": round(parallel_total_ms / len(parallel_configs), 1),
        },
        "session_stats": session.stats.__dict__,
        "correctness": {
            "warm_min_sharpe": ref_min_sharpe,
            "expected_min_sharpe": 0.4742679235329245,
            "match": abs(ref_min_sharpe - 0.4742679235329245) < 1e-6,
        },
        "talking_points": [
            f"Pre-optimization: ~25s per multi-window eval (subprocess + 6x backtest)",
            f"Post-optimization cold CLI: {cold_ms/1000:.1f}s per config",
            f"Warm EvalSession eval: {warm_eval_ms:.0f}ms per config ({24899/warm_eval_ms:.0f}x vs original)",
            f"25-config autoresearch loop: ~{25 * warm_eval_ms / 1000:.0f}s in-process vs ~{25 * 24899 / 1000:.0f}s originally",
            f"Research sweep throughput: {1000/sweep_per_config_ms:.1f} configs/sec ({n_configs} configs in {sweep_total_ms/1000:.1f}s)",
            "EvalSession: load panel+cache once, sweep in-process — standard quant research platform pattern",
            "Single-pass backtest + period slice (evaluate_multiwindow_fast) avoids redundant simulation",
        ],
        "parallel_sample": parallel_results,
        "sweep_best": max(sweep_results, key=lambda m: m["min_sharpe"]),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
