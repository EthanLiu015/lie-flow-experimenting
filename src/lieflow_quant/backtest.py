"""Cross-sectional long/short backtest for LieFlow trading signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    gross_exposure: pd.Series
    turnover: pd.Series
    metrics: dict


def cross_sectional_weights(
    day_signals: pd.Series,
    gross_exposure: float,
) -> pd.Series:
    """Dollar-neutral weights from cross-sectional signal ranks."""
    ranks = day_signals.rank(method="average", pct=True)
    centered = ranks - ranks.mean()
    denom = centered.abs().sum()
    if denom < 1e-12:
        return pd.Series(0.0, index=day_signals.index)
    return centered / denom * gross_exposure


def information_coefficient(
    signals: pd.Series,
    forward_returns: pd.Series,
) -> float:
    aligned = pd.concat([signals, forward_returns], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def summarize_returns(daily: pd.Series) -> dict:
    daily = daily.dropna()
    if daily.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    equity = (1 + daily).cumprod()
    total = float(equity.iloc[-1] - 1)
    ann_factor = 252
    ann_ret = float((1 + total) ** (ann_factor / len(daily)) - 1) if len(daily) else 0.0
    ann_vol = float(daily.std() * np.sqrt(ann_factor))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    return {
        "total_return": total,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
    }


def run_backtest(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    lag: int = 1,
    cost_bps: float = 5.0,
    signal_col: str = "signal",
    hold_days: int = 1,
) -> BacktestResult:
    """
    Backtest canonical-residual L/S with symmetry concentration overlay.

    ``signals`` must have columns: date, ticker, signal, gross_exposure.
    ``forward_returns`` is a wide DataFrame indexed by date with ticker columns.
    """
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])

    dates = sorted(sig["date"].unique())
    prev_weights: pd.Series | None = None
    daily_rets: list[float] = []
    daily_dates: list[pd.Timestamp] = []
    gross_series: list[float] = []
    turnover_series: list[float] = []
    ic_list: list[float] = []

    for i, date in enumerate(dates):
        day = sig[sig["date"] == date]
        if day.empty:
            continue

        gross = float(day["gross_exposure"].iloc[0])
        w = cross_sectional_weights(
            day.set_index("ticker")[signal_col],
            gross,
        )

        if hold_days > 1 and prev_weights is not None:
            w = (w + prev_weights) / 2

        trade_date = date + pd.tseries.offsets.BDay(lag)
        if trade_date not in forward_returns.index:
            continue

        fwd = forward_returns.loc[trade_date].reindex(w.index)
        ic_list.append(information_coefficient(day.set_index("ticker")[signal_col], fwd))

        port_ret = float((w * fwd.fillna(0)).sum())
        turnover = 0.0 if prev_weights is None else float((w - prev_weights.reindex(w.index).fillna(0)).abs().sum())
        cost = turnover * cost_bps / 10_000
        net_ret = port_ret - cost

        daily_rets.append(net_ret)
        daily_dates.append(trade_date)
        gross_series.append(gross)
        turnover_series.append(turnover)
        prev_weights = w

    daily = pd.Series(daily_rets, index=pd.DatetimeIndex(daily_dates), name="return")
    metrics = summarize_returns(daily)
    metrics["mean_ic"] = float(np.nanmean(ic_list)) if ic_list else float("nan")
    metrics["mean_turnover"] = float(np.mean(turnover_series)) if turnover_series else 0.0
    metrics["mean_gross_exposure"] = float(np.mean(gross_series)) if gross_series else 0.0
    metrics["n_days"] = len(daily)

    return BacktestResult(
        daily_returns=daily,
        gross_exposure=pd.Series(gross_series, index=daily.index),
        turnover=pd.Series(turnover_series, index=daily.index),
        metrics=metrics,
    )


def run_momentum_benchmark(
    close: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    signal_dates: pd.DatetimeIndex | None = None,
    momentum_window: int = 20,
    lag: int = 1,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Vanilla cross-sectional momentum L/S on the same universe."""
    px = close[[c for c in close.columns if c != "^VIX"]]
    mom = np.log(px / px.shift(1)).rolling(momentum_window).sum()

    if signal_dates is not None:
        dates = pd.DatetimeIndex(signal_dates).normalize().unique()
        mom = mom.loc[mom.index.intersection(dates)]

    long_rows = []
    for date in mom.index:
        vals = mom.loc[date].dropna()
        if len(vals) < 10:
            continue
        for ticker, val in vals.items():
            long_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "signal": float(val),
                    "gross_exposure": 1.0,
                }
            )

    bench_signals = pd.DataFrame(long_rows)
    return run_backtest(
        bench_signals,
        forward_returns,
        lag=lag,
        cost_bps=cost_bps,
    )
