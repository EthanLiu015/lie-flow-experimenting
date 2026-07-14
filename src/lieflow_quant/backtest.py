"""Cross-sectional long/short backtest for LieFlow trading signals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lieflow_quant.methodology import calendar_trade_dates


def _average_ranks(vals: np.ndarray) -> np.ndarray:
    """Average ranks for 1-D values (ties get mean rank)."""
    n = len(vals)
    if n == 0:
        return vals.astype(np.float64)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_v = vals[order]
    start = 0
    while start < n:
        end = start
        while end + 1 < n and sorted_v[end + 1] == sorted_v[start]:
            end += 1
        if end > start:
            avg = (start + end + 2) / 2.0
            ranks[order[start : end + 1]] = avg
        start = end + 1
    return ranks


def _spearman_ic(signals: np.ndarray, forward: np.ndarray) -> float:
    mask = np.isfinite(signals) & np.isfinite(forward)
    n = int(mask.sum())
    if n < 5:
        return float("nan")
    rs = _average_ranks(signals[mask])
    rf = _average_ranks(forward[mask])
    rs -= rs.mean()
    rf -= rf.mean()
    denom = float(np.sqrt((rs * rs).sum() * (rf * rf).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((rs * rf).sum() / denom)


def _cross_sectional_weights_np(signals: np.ndarray, gross_exposure: float) -> np.ndarray:
    if len(signals) == 0:
        return signals
    ranks = _average_ranks(signals) / len(signals)
    centered = ranks - ranks.mean()
    denom = float(np.abs(centered).sum())
    if denom < 1e-12:
        return np.zeros_like(signals)
    return centered / denom * gross_exposure


def _cross_sectional_weights_topbottom(
    signals: np.ndarray,
    gross_exposure: float,
    long_n: int,
    short_n: int,
) -> np.ndarray:
    """Weights using only the top long_n and bottom short_n by signal rank."""
    n = len(signals)
    if n == 0:
        return signals
    order = np.argsort(signals, kind="stable")
    keep = np.zeros(n, dtype=bool)
    keep[order[:short_n]] = True
    keep[order[-long_n:]] = True
    filtered = np.where(keep, signals, np.nan)
    valid_mask = ~np.isnan(filtered)
    if valid_mask.sum() < 2:
        return np.zeros(n)
    valid_vals = filtered[valid_mask]
    ranks = _average_ranks(valid_vals) / len(valid_vals)
    centered = ranks - ranks.mean()
    denom = float(np.abs(centered).sum())
    if denom < 1e-12:
        return np.zeros(n)
    w_valid = centered / denom * gross_exposure
    out = np.zeros(n)
    out[valid_mask] = w_valid
    return out


def _turnover_dict(current: dict[str, float], previous: dict[str, float] | None) -> float:
    if previous is None:
        return 0.0
    keys = current.keys() | previous.keys()
    return float(sum(abs(current.get(k, 0.0) - previous.get(k, 0.0)) for k in keys))


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    gross_exposure: pd.Series
    turnover: pd.Series
    metrics: dict
    daily_ic: pd.Series | None = None


def dedupe_day_signals(day: pd.DataFrame, signal_col: str = "signal") -> pd.Series:
    """Average duplicate tickers (e.g. from cloud padding) before weighting."""
    grouped = day.groupby("ticker", as_index=True)[signal_col].mean()
    return grouped


def information_coefficient(
    signals: pd.Series,
    forward_returns: pd.Series,
) -> float:
    return _spearman_ic(
        signals.to_numpy(dtype=float, copy=False),
        forward_returns.to_numpy(dtype=float, copy=False),
    )


def cross_sectional_weights(
    day_signals: pd.Series,
    gross_exposure: float,
) -> pd.Series:
    """Dollar-neutral weights from cross-sectional signal ranks."""
    vals = day_signals.to_numpy(dtype=float, copy=False)
    idx = day_signals.index
    if len(vals) == 0:
        return pd.Series(0.0, index=idx)
    weights = _cross_sectional_weights_np(vals, gross_exposure)
    return pd.Series(weights, index=idx)


def blend_hold_weights(
    target_weights: pd.Series,
    weight_history: deque[pd.Series],
    hold_days: int,
) -> pd.Series:
    """Average target weights over the trailing ``hold_days`` rebalance dates."""
    history = list(weight_history)
    history.append(target_weights)
    if hold_days <= 1 or len(history) == 1:
        return target_weights

    window = history[-hold_days:]
    idx = window[-1].index
    for prior in window[:-1]:
        idx = idx.union(prior.index)

    blended = pd.Series(0.0, index=sorted(idx))
    for prior in window:
        blended = blended.add(prior.reindex(blended.index, fill_value=0.0), fill_value=0.0)
    return blended / len(window)


def portfolio_return(
    weights: pd.Series,
    forward: pd.Series,
    gross_exposure: float,
    *,
    min_names: int = 5,
) -> tuple[float, pd.Series, pd.Series]:
    """
    Compute portfolio return excluding names with missing forward returns.

    Rescales weights to preserve gross exposure on the tradable subset.
    """
    w = weights.to_numpy(dtype=float, copy=False)
    f = forward.reindex(weights.index).to_numpy(dtype=float, copy=False)
    mask = np.isfinite(w) & np.isfinite(f)
    n = int(mask.sum())
    if n < min_names:
        return float("nan"), pd.Series(dtype=float), pd.Series(dtype=float)

    w = w[mask]
    f = f[mask]
    tickers = weights.index[mask]
    gross = float(np.abs(w).sum())
    if gross < 1e-12:
        return 0.0, pd.Series(dtype=float), pd.Series(dtype=float)

    w = w / gross * gross_exposure
    w_series = pd.Series(w, index=tickers)
    f_series = pd.Series(f, index=tickers)
    return float((w * f).sum()), w_series, f_series


def turnover_between(
    current: pd.Series,
    previous: pd.Series | None,
) -> float:
    if previous is None:
        return 0.0
    return _turnover_dict(current.to_dict(), previous.to_dict())


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
    sharpe = float(daily.mean() / daily.std() * np.sqrt(ann_factor)) if daily.std() > 1e-12 else 0.0
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
    cost_bps: float = 10.0,
    signal_col: str = "signal",
    hold_days: int = 1,
    dedupe_tickers: bool = True,
    forward_horizon: int = 1,
    fill_calendar: bool = True,
    long_n: int = 0,
    short_n: int = 0,
) -> BacktestResult:
    """
    Backtest canonical-residual L/S with symmetry concentration overlay.

    ``signals`` must have columns: date, ticker, signal, gross_exposure.
    ``forward_returns`` is a wide DataFrame indexed by date with ticker columns.
    """
    sig = signals
    if not pd.api.types.is_datetime64_any_dtype(sig["date"]):
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"])

    if dedupe_tickers:
        sig = sig.groupby(["date", "ticker"], as_index=False).agg(
            {signal_col: "mean", "gross_exposure": "first"}
        )

    fwd_values = forward_returns.to_numpy(dtype=float, copy=False)
    fwd_index = forward_returns.index
    fwd_pos = {pd.Timestamp(d): i for i, d in enumerate(fwd_index)}
    col_pos = {str(c): i for i, c in enumerate(forward_returns.columns)}
    bday = pd.tseries.offsets.BDay(lag)
    prev_weights: dict[str, float] | None = None
    prev_weights_pd: pd.Series | None = None
    weight_history: deque[pd.Series] = deque(maxlen=max(hold_days, 1))
    daily_rets: list[float] = []
    daily_dates: list[pd.Timestamp] = []
    gross_series: list[float] = []
    turnover_series: list[float] = []
    ic_list: list[float] = []
    ic_dates: list[pd.Timestamp] = []
    use_fast_path = hold_days <= 1
    _use_topbottom = long_n > 0 and short_n > 0

    for date, day in sig.groupby("date", sort=True):
        if day.empty:
            continue

        gross = float(day["gross_exposure"].iloc[0])
        tickers = day["ticker"].astype(str).to_numpy()
        sig_vals = day[signal_col].to_numpy(dtype=float, copy=False)

        if use_fast_path:
            trade_date = pd.Timestamp(date) + bday
            row_i = fwd_pos.get(trade_date)
            if row_i is None:
                continue

            if gross < 1e-12:
                w_dict: dict[str, float] = {}
                turnover = _turnover_dict(w_dict, prev_weights)
                net_ret = -turnover * cost_bps / 10_000
                daily_rets.append(net_ret)
                daily_dates.append(trade_date)
                gross_series.append(0.0)
                turnover_series.append(turnover)
                prev_weights = w_dict
                continue

            target_w = (
                _cross_sectional_weights_topbottom(sig_vals, gross, long_n, short_n)
                if _use_topbottom
                else _cross_sectional_weights_np(sig_vals, gross)
            )
            fwd_row = fwd_values[row_i]
            col_idx = np.fromiter((col_pos.get(t, -1) for t in tickers), dtype=np.int64, count=len(tickers))
            valid = col_idx >= 0
            if not valid.any():
                continue
            w = target_w[valid]
            f = fwd_row[col_idx[valid]]
            mask = np.isfinite(w) & np.isfinite(f)
            n = int(mask.sum())
            if n < 5:
                continue
            w = w[mask]
            f = f[mask]
            used_tickers = tickers[valid][mask]
            sig_used = sig_vals[valid][mask]
            gross_w = float(np.abs(w).sum())
            if gross_w < 1e-12:
                continue
            w = w / gross_w * gross
            port_ret = float((w * f).sum())
            w_dict = dict(zip(used_tickers, w, strict=True))
            ic_list.append(_spearman_ic(sig_used, f))
            ic_dates.append(trade_date)
            turnover = _turnover_dict(w_dict, prev_weights)
            net_ret = port_ret - turnover * cost_bps / 10_000
            daily_rets.append(net_ret)
            daily_dates.append(trade_date)
            gross_series.append(gross)
            turnover_series.append(turnover)
            prev_weights = w_dict
            continue

        day_signals = pd.Series(sig_vals, index=tickers)
        target_w = cross_sectional_weights(day_signals, gross)
        w = blend_hold_weights(target_w, weight_history, hold_days)

        trade_date = pd.Timestamp(date) + bday
        if trade_date not in fwd_index:
            continue

        fwd = forward_returns.loc[trade_date].reindex(w.index)
        port_ret, w_used, fwd_used = portfolio_return(w, fwd, gross)
        if np.isnan(port_ret):
            continue

        weight_history.append(target_w)
        ic_list.append(information_coefficient(day_signals.reindex(w_used.index), fwd_used))
        ic_dates.append(trade_date)
        turnover = turnover_between(w, prev_weights_pd)
        net_ret = port_ret - turnover * cost_bps / 10_000

        daily_rets.append(net_ret)
        daily_dates.append(trade_date)
        gross_series.append(gross)
        turnover_series.append(turnover)
        prev_weights_pd = w

    daily = pd.Series(daily_rets, index=pd.DatetimeIndex(daily_dates), name="return")
    gross_s = pd.Series(gross_series, index=pd.DatetimeIndex(daily_dates), name="gross") if gross_series else pd.Series(dtype=float)
    turnover_s = (
        pd.Series(turnover_series, index=pd.DatetimeIndex(daily_dates), name="turnover")
        if turnover_series
        else pd.Series(dtype=float)
    )
    if fill_calendar and not daily.empty:
        cal = calendar_trade_dates(sig["date"], forward_returns.index, lag=lag)
        daily = daily.reindex(cal, fill_value=0.0)
        gross_s = gross_s.reindex(cal, fill_value=0.0)
        turnover_s = turnover_s.reindex(cal, fill_value=0.0)

    daily_ic = (
        pd.Series(ic_list, index=pd.DatetimeIndex(ic_dates), name="ic")
        if ic_list
        else None
    )
    metrics = summarize_returns(daily)
    metrics["mean_ic"] = float(np.nanmean(ic_list)) if ic_list else float("nan")
    metrics["mean_turnover"] = float(turnover_s.mean()) if not turnover_s.empty else 0.0
    metrics["mean_gross_exposure"] = float(gross_s.mean()) if not gross_s.empty else 0.0
    metrics["n_days"] = len(daily)
    metrics["n_trade_days"] = int((gross_s > 1e-12).sum()) if not gross_s.empty else 0
    metrics["forward_horizon"] = forward_horizon

    return BacktestResult(
        daily_returns=daily,
        gross_exposure=gross_s,
        turnover=turnover_s,
        metrics=metrics,
        daily_ic=daily_ic,
    )


def run_momentum_benchmark(
    close: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    signal_dates: pd.DatetimeIndex | None = None,
    universe_by_date: dict[pd.Timestamp, list[str]] | None = None,
    momentum_window: int = 20,
    lag: int = 1,
    cost_bps: float = 10.0,
    dedupe_tickers: bool = True,
    forward_horizon: int = 1,
    hold_days: int = 1,
    fill_calendar: bool = True,
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
        if universe_by_date is not None:
            key = pd.Timestamp(date).normalize()
            allowed = universe_by_date.get(key)
            if allowed:
                vals = vals.reindex(allowed).dropna()
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
        dedupe_tickers=dedupe_tickers,
        forward_horizon=forward_horizon,
        hold_days=hold_days,
        fill_calendar=fill_calendar,
    )


def build_raw_vol_signals(
    close: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    *,
    universe_by_date: dict[pd.Timestamp, list[str]] | None = None,
    vol_window: int = 20,
    signal_smoothing: int = 1,
    signal_sign: float = 1.0,
) -> pd.DataFrame:
    """Baseline high/low vol cross-section on the same dates and universe."""
    px = close[[c for c in close.columns if c != "^VIX"]]
    log_ret = np.log(px / px.shift(1))
    raw_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

    dates = pd.DatetimeIndex(signal_dates).normalize().unique()
    dates = dates.intersection(raw_vol.index)
    if len(dates) == 0:
        return pd.DataFrame(columns=["date", "ticker", "signal", "gross_exposure"])

    long = (
        raw_vol.loc[dates]
        .stack(future_stack=True)
        .rename("signal")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )
    long = long.dropna(subset=["signal"])
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()

    if universe_by_date is not None:
        keys = pd.MultiIndex.from_arrays(
            [
                np.repeat(list(universe_by_date.keys()), [len(v) for v in universe_by_date.values()]),
                [t for tickers in universe_by_date.values() for t in tickers],
            ],
            names=["date", "ticker"],
        )
        keys = keys.set_levels(pd.to_datetime(keys.levels[0]).normalize(), level=0)
        long = long.set_index(["date", "ticker"]).loc[long.set_index(["date", "ticker"]).index.intersection(keys)].reset_index()

    counts = long.groupby("date")["ticker"].transform("count")
    long = long[counts >= 10].copy()
    if long.empty:
        return pd.DataFrame(columns=["date", "ticker", "signal", "gross_exposure"])

    long["signal"] = long["signal"].astype(float) * signal_sign
    long["gross_exposure"] = 1.0

    if signal_smoothing > 1:
        long = long.sort_values(["ticker", "date"])
        long["signal"] = long.groupby("ticker")["signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )

    return long[["date", "ticker", "signal", "gross_exposure"]].copy()
