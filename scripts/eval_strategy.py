#!/usr/bin/env python3
"""Fast strategy evaluation from cached LieFlow inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lieflow_quant.backtest import (
    build_raw_vol_signals,
    run_backtest,
    run_momentum_benchmark,
)
from lieflow_quant.cache import build_signals_from_cache, load_inference_cache
from lieflow_quant.panel import compute_forward_returns, load_equity_panel


def _metric_subset(result_metrics: dict) -> dict:
    keys = (
        "sharpe",
        "mean_ic",
        "n_days",
        "total_return",
        "mean_turnover",
    )
    return {k: result_metrics[k] for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LieFlow strategy from inference cache.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/strategy/inference_cache.npz"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/equity"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/strategy/eval_metrics.json"),
    )
    parser.add_argument("--signal-feature", default="canonical_momentum")
    parser.add_argument("--signal-sign", type=float, default=1.0)
    parser.add_argument("--concentration-window", type=int, default=60)
    parser.add_argument("--min-exposure", type=float, default=0.25)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--signal-smoothing", type=int, default=1)
    parser.add_argument("--regime-filter", default=None)
    parser.add_argument("--combine-concentration", action="store_true")
    parser.add_argument("--dedupe-tickers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--hold-days", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--forward-horizon", type=int, default=1)
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--include-benchmark", action="store_true")
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Also report raw-vol baseline on the same universe/dates.",
    )
    args = parser.parse_args()

    if not args.cache.exists():
        raise FileNotFoundError(
            f"Cache not found: {args.cache}\nRun scripts/cache_inference.py first."
        )

    cache = load_inference_cache(args.cache)
    signals = build_signals_from_cache(
        cache,
        signal_feature=args.signal_feature,
        signal_sign=args.signal_sign,
        concentration_window=args.concentration_window,
        min_exposure=args.min_exposure,
        max_exposure=args.max_exposure,
        signal_smoothing=args.signal_smoothing,
        regime_filter=args.regime_filter,
        combine_concentration=args.combine_concentration,
        dedupe_tickers=args.dedupe_tickers,
    )

    close, _ = load_equity_panel(args.data_dir)
    fwd = compute_forward_returns(close, horizon=args.forward_horizon)

    universe_by_date = (
        signals.groupby("date")["ticker"].apply(list).to_dict()
        if args.dedupe_tickers
        else None
    )

    result = run_backtest(
        signals,
        fwd,
        lag=args.lag,
        cost_bps=args.cost_bps,
        hold_days=args.hold_days,
        dedupe_tickers=args.dedupe_tickers,
        forward_horizon=args.forward_horizon,
    )

    metrics = {
        "sharpe": result.metrics["sharpe"],
        "mean_ic": result.metrics["mean_ic"],
        "n_days": result.metrics["n_days"],
        "total_return": result.metrics["total_return"],
        "annualized_return": result.metrics["annualized_return"],
        "annualized_vol": result.metrics["annualized_vol"],
        "max_drawdown": result.metrics["max_drawdown"],
        "mean_turnover": result.metrics["mean_turnover"],
        "mean_gross_exposure": result.metrics["mean_gross_exposure"],
        "signal_feature": args.signal_feature,
        "signal_sign": args.signal_sign,
        "lag": args.lag,
        "hold_days": args.hold_days,
        "cost_bps": args.cost_bps,
        "forward_horizon": args.forward_horizon,
        "signal_smoothing": args.signal_smoothing,
        "regime_filter": args.regime_filter,
        "combine_concentration": args.combine_concentration,
        "dedupe_tickers": args.dedupe_tickers,
    }

    if args.include_benchmark:
        bench = run_momentum_benchmark(
            close,
            fwd,
            signal_dates=signals["date"],
            universe_by_date=universe_by_date,
            momentum_window=args.momentum_window,
            lag=args.lag,
            cost_bps=args.cost_bps,
            dedupe_tickers=args.dedupe_tickers,
            forward_horizon=args.forward_horizon,
            hold_days=args.hold_days,
        )
        metrics["benchmark_sharpe"] = bench.metrics["sharpe"]
        metrics["benchmark_mean_ic"] = bench.metrics["mean_ic"]

    if args.include_ablations:
        raw_vol = build_raw_vol_signals(
            close,
            signals["date"],
            universe_by_date=universe_by_date,
            signal_smoothing=args.signal_smoothing,
            signal_sign=args.signal_sign,
        )
        raw_result = run_backtest(
            raw_vol,
            fwd,
            lag=args.lag,
            cost_bps=args.cost_bps,
            hold_days=args.hold_days,
            forward_horizon=args.forward_horizon,
        )
        metrics["ablation_raw_vol"] = _metric_subset(raw_result.metrics)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
