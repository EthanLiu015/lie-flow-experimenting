#!/usr/bin/env python3
"""Backtest LieFlow canonical-residual L/S vs momentum benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lieflow_quant.backtest import run_backtest, run_momentum_benchmark
from lieflow_quant.panel import compute_forward_returns, load_equity_panel


def plot_equity_curves(results: dict[str, pd.Series], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, daily in results.items():
        equity = (1 + daily).cumprod()
        ax.plot(equity.index, equity.values, label=name)
    ax.set_title("Cumulative return (close-to-close, 1-day lag)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signals",
        type=Path,
        default=Path("outputs/strategy/daily_signals.csv"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/equity"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/strategy"))
    parser.add_argument("--lag", type=int, default=1, help="Trade lag in business days.")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--hold-days", type=int, default=1)
    parser.add_argument("--forward-horizon", type=int, default=1)
    args = parser.parse_args()

    if not args.signals.exists():
        raise FileNotFoundError(
            f"Signals not found: {args.signals}\n"
            "Run scripts/generate_trading_signals.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(args.signals, parse_dates=["date"])
    close, _ = load_equity_panel(args.data_dir)
    fwd = compute_forward_returns(close, horizon=args.forward_horizon)

    lieflow = run_backtest(
        signals,
        fwd,
        lag=args.lag,
        cost_bps=args.cost_bps,
        hold_days=args.hold_days,
        dedupe_tickers=True,
        forward_horizon=args.forward_horizon,
    )
    bench = run_momentum_benchmark(
        close,
        fwd,
        signal_dates=signals["date"],
        universe_by_date=signals.groupby("date")["ticker"].apply(list).to_dict(),
        lag=args.lag,
        cost_bps=args.cost_bps,
        hold_days=args.hold_days,
        forward_horizon=args.forward_horizon,
    )

    summary = {
        "lieflow_canonical_momentum": lieflow.metrics,
        "momentum_benchmark": bench.metrics,
    }
    with open(args.output_dir / "backtest_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    lieflow.daily_returns.to_csv(args.output_dir / "lieflow_daily_returns.csv")
    bench.daily_returns.to_csv(args.output_dir / "momentum_daily_returns.csv")

    plot_equity_curves(
        {
            "LieFlow canonical L/S": lieflow.daily_returns,
            "Momentum L/S": bench.daily_returns,
        },
        args.output_dir / "equity_curves.png",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved results -> {args.output_dir}")


if __name__ == "__main__":
    main()
