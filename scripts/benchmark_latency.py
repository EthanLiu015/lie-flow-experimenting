#!/usr/bin/env python3
"""Benchmark lieflow_quant pipeline latency (ms)."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/tmp/bench_multiwindow.json")

from lieflow_quant.backtest import run_backtest
from lieflow_quant.cache import build_signals_from_cache, load_inference_cache
from lieflow_quant.panel import (
    build_daily_cross_sections,
    compute_forward_returns,
    load_equity_panel,
)
from lieflow_quant.signals_advanced import build_advanced_signals, build_hybrid_vol_concentration_signals
from lieflow_quant.validation import (
    DEFAULT_PERIODS,
    build_panel_sections,
    build_raw_vol_signals_for_sections,
    evaluate_signals_in_period,
)


@dataclass
class BenchResult:
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        s = sorted(self.times_ms)
        return s[len(s) // 2]

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms)


@contextmanager
def timed(results: dict[str, BenchResult], name: str):
    t0 = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - t0) * 1000
    results.setdefault(name, BenchResult(name)).times_ms.append(elapsed)


def run_benchmark(
    *,
    data_dir: Path,
    cache_path: Path,
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, float]:
    results: dict[str, BenchResult] = {}

    for _ in range(warmup):
        close, vix = load_equity_panel(data_dir)
        cache = load_inference_cache(cache_path)
        sections = build_panel_sections(close, n_target=60, vix=vix)
        fwd = compute_forward_returns(close)
        signals = build_signals_from_cache(cache, signal_smoothing=20)
        adv = build_advanced_signals(cache, strategy="mom_minus_vol", signal_sign=-1, signal_smoothing=25)
        raw = build_raw_vol_signals_for_sections(close, sections, signal_smoothing=20)
        hybrid = build_hybrid_vol_concentration_signals(cache, raw, signal_smoothing=20, min_concentration_ratio=1.0)
        for period in DEFAULT_PERIODS:
            evaluate_signals_in_period(signals, close, period, cost_bps=10.0)
        run_backtest(signals, fwd, cost_bps=10.0)

    for _ in range(repeats):
        with timed(results, "load_equity_panel"):
            close, vix = load_equity_panel(data_dir)

        with timed(results, "load_inference_cache"):
            cache = load_inference_cache(cache_path)

        with timed(results, "build_panel_sections"):
            sections = build_panel_sections(close, n_target=60, vix=vix)

        with timed(results, "build_daily_cross_sections"):
            build_daily_cross_sections(close, n_target=60, vix=vix)

        with timed(results, "compute_forward_returns"):
            fwd = compute_forward_returns(close)

        with timed(results, "build_signals_from_cache"):
            signals = build_signals_from_cache(cache, signal_smoothing=20)

        with timed(results, "build_advanced_signals"):
            build_advanced_signals(cache, strategy="mom_minus_vol", signal_sign=-1, signal_smoothing=25)

        with timed(results, "build_raw_vol_signals"):
            raw = build_raw_vol_signals_for_sections(close, sections, signal_smoothing=20)

        with timed(results, "build_hybrid_vol_signals"):
            build_hybrid_vol_concentration_signals(
                cache, raw, signal_smoothing=20, min_concentration_ratio=1.0
            )

        with timed(results, "run_backtest_single"):
            run_backtest(signals, fwd, cost_bps=10.0)

        with timed(results, "eval_multiwindow_script"):
            import subprocess
            import sys as _sys

            subprocess.run(
                [
                    _sys.executable,
                    str(ROOT / "scripts" / "eval_multiwindow.py"),
                    "--raw-vol-baseline",
                    "--signal-smoothing",
                    "20",
                    "--output-json",
                    str(OUT),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )

        with timed(results, "full_eval_pipeline"):
            close2, vix2 = load_equity_panel(data_dir)
            cache2 = load_inference_cache(cache_path)
            build_panel_sections(close2, n_target=60, vix=vix2)
            sig2 = build_advanced_signals(
                cache2, strategy="mom_minus_vol", signal_sign=-1, signal_smoothing=25
            )
            fwd2 = compute_forward_returns(close2)
            for period in DEFAULT_PERIODS:
                evaluate_signals_in_period(
                    sig2, close2, period, cost_bps=10.0, forward_returns=fwd2
                )

    return {name: round(r.median_ms, 2) for name, r in sorted(results.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark lieflow_quant latency.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/equity"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/strategy/inference_cache_full_n60.npz"),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    if not args.cache.exists():
        raise FileNotFoundError(f"Cache not found: {args.cache}")

    metrics = run_benchmark(
        data_dir=args.data_dir,
        cache_path=args.cache,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    total = round(sum(metrics.values()), 2)
    out = {"label": args.label, "metrics_ms": metrics, "sum_ms": total}
    print(json.dumps(out, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
