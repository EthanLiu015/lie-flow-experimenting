#!/usr/bin/env python3
"""Download equity panel and VIX for cross-section cloud construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

# Liquid S&P 100 subset (ticker, name) — stable large caps for daily panels.
TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "UNH", "XOM", "V", "MA", "PG", "JNJ", "HD", "COST", "ABBV", "MRK",
    "CVX", "KO", "PEP", "WMT", "BAC", "CRM", "AMD", "NFLX", "TMO", "ADBE",
    "LIN", "MCD", "CSCO", "ACN", "ABT", "DHR", "WFC", "TXN", "DIS", "PM",
    "INTC", "VZ", "CMCSA", "IBM", "QCOM", "ORCL", "INTU", "AMAT", "GE", "CAT",
    "NOW", "ISRG", "BKNG", "GS", "AXP", "MS", "BLK", "SPGI", "RTX", "HON",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/equity"))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(TICKERS)} tickers + ^VIX ...")
    prices = yf.download(
        TICKERS + ["^VIX"],
        start=args.start,
        end=args.end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"]
    else:
        close = prices

    close = close.dropna(how="all")
    close.to_csv(args.output_dir / "close_prices.csv")
    print(f"Saved close prices: {close.shape} -> {args.output_dir / 'close_prices.csv'}")

    vix = close["^VIX"].rename("vix").dropna()
    vix.to_frame().to_csv(args.output_dir / "vix.csv")
    print(f"Saved VIX series: {len(vix)} days")


if __name__ == "__main__":
    main()
