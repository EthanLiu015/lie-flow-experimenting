"""Shared methodology helpers for honest backtesting."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def ml_temporal_test_start(
    dates: Iterable[pd.Timestamp],
    *,
    train_ratio: float = 0.8,
) -> pd.Timestamp:
    """
    First calendar date in the LieFlow model *test* split (temporal 80/20).

    Matches ``CrossSectionCloudDataset`` with ``split_mode=temporal``.
    """
    sorted_dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).unique().sort_values()
    if len(sorted_dates) == 0:
        raise ValueError("no dates provided for ML split")
    if len(sorted_dates) == 1:
        return pd.Timestamp(sorted_dates[0]).normalize()
    split_idx = int(len(sorted_dates) * train_ratio)
    split_idx = min(max(split_idx, 1), len(sorted_dates) - 1)
    return pd.Timestamp(sorted_dates[split_idx]).normalize()


def concentration_ratio_series(
    daily_conc: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Today's concentration divided by the trailing median of *prior* days only."""
    daily_conc = daily_conc.sort_index()
    med = daily_conc.shift(1).rolling(window, min_periods=1).median().clip(lower=1e-6)
    return daily_conc / med


def calendar_trade_dates(
    signal_dates: pd.DatetimeIndex | pd.Series,
    forward_index: pd.DatetimeIndex,
    *,
    lag: int = 1,
) -> pd.DatetimeIndex:
    """Business-day trade dates implied by signal dates and execution lag."""
    if len(signal_dates) == 0:
        return pd.DatetimeIndex([])
    bday = pd.tseries.offsets.BDay(lag)
    sig = pd.DatetimeIndex(pd.to_datetime(signal_dates)).normalize().unique().sort_values()
    start = pd.Timestamp(sig[0]) + bday
    end = pd.Timestamp(sig[-1]) + bday
    return forward_index[(forward_index >= start) & (forward_index <= end)]


def _is_nan(x: object) -> bool:
    return isinstance(x, float) and x != x


def effective_period_sharpe(pm: dict) -> float:
    """
    Sharpe for constraint metrics: flat/no-trade windows (NaN) count as 0.

    Backtest sets Sharpe=NaN when daily return volatility is zero; for portfolio-level
    min/mean Sharpe across windows we treat those as 0, not excluded.
    """
    sh = pm.get("sharpe")
    tr = float(pm.get("total_return", 0.0))
    if sh is None or _is_nan(sh):
        return 0.0 if tr == 0.0 else float("nan")
    return float(sh)


def period_annualized_return(pm: dict) -> float:
    """Annualized return from period total return and day count."""
    n = int(pm.get("n_days", 0))
    if n <= 0:
        return 0.0
    tr = float(pm.get("total_return", 0.0))
    return float((1.0 + tr) ** (252.0 / n) - 1.0)


def aggregate_multiwindow_metrics(
    period_metrics: dict[str, dict],
    period_names: tuple[str, ...],
) -> dict[str, float]:
    """
    Honest aggregates across all configured evaluation windows.

    Every period in ``period_names`` contributes; missing or empty periods are 0.
    """
    sharpes: list[float] = []
    total_rets: list[float] = []
    ann_rets: list[float] = []

    for name in period_names:
        pm = period_metrics.get(name)
        if not isinstance(pm, dict) or pm.get("n_days", 0) <= 0:
            sharpes.append(0.0)
            total_rets.append(0.0)
            ann_rets.append(0.0)
            continue
        sharpes.append(effective_period_sharpe(pm))
        total_rets.append(float(pm.get("total_return", 0.0)))
        ann_rets.append(period_annualized_return(pm))

    if not period_names:
        nan = float("nan")
        return {
            "min_sharpe": nan,
            "mean_sharpe": nan,
            "mean_total_return": nan,
            "min_total_return": nan,
            "mean_annualized_return": nan,
            "n_periods_positive_sharpe": 0,
        }

    return {
        "min_sharpe": min(sharpes),
        "mean_sharpe": sum(sharpes) / len(sharpes),
        "mean_total_return": sum(total_rets) / len(total_rets),
        "min_total_return": min(total_rets),
        "mean_annualized_return": sum(ann_rets) / len(ann_rets),
        "n_periods_positive_sharpe": sum(1 for s in sharpes if s > 0),
    }


def lieflow_influence_fraction(
    signals: pd.DataFrame,
    lieflow_dates: Iterable[pd.Timestamp],
    *,
    gross_col: str = "gross_exposure",
    fallback_gross: float = 1.0,
    tol: float = 1e-6,
) -> float:
    """
    Fraction of LieFlow-available dates where exposure differs from raw-vol fallback.

    Used to reject configs that ignore LieFlow sizing on OOS days.
    """
    lie_set = {pd.Timestamp(d).normalize() for d in lieflow_dates}
    if not lie_set or signals.empty:
        return 0.0
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    daily_gross = sig.groupby("date")[gross_col].first()
    influenced = 0
    for d in lie_set:
        if d not in daily_gross.index:
            continue
        g = float(daily_gross.loc[d])
        if abs(g - fallback_gross) > tol:
            influenced += 1
    return influenced / len(lie_set)


def lieflow_alpha_influence_fraction(
    signals: pd.DataFrame,
    raw_vol_signals: pd.DataFrame,
    lieflow_dates: Iterable[pd.Timestamp],
    *,
    tol: float = 1e-8,
) -> float:
    """
    Fraction of LieFlow-available (date, ticker) rows where blended signal != raw vol.

    Confirms LieFlow alpha is actually changing cross-sectional rankings.
    """
    lie_set = {pd.Timestamp(d).normalize() for d in lieflow_dates}
    if not lie_set or signals.empty:
        return 0.0
    sig = signals.copy()
    raw = raw_vol_signals.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    merged = sig.merge(
        raw[["date", "ticker", "signal"]].rename(columns={"signal": "raw_signal"}),
        on=["date", "ticker"],
        how="inner",
    )
    merged = merged[merged["date"].isin(lie_set)]
    if merged.empty:
        return 0.0
    diff = (merged["signal"] - merged["raw_signal"]).abs() > tol
    return float(diff.sum()) / len(merged)


def coverage_fraction(
    signals: pd.DataFrame,
    universe_dates: Iterable[pd.Timestamp],
    *,
    signal_col: str = "signal",
    gross_col: str = "gross_exposure",
) -> float:
    """Fraction of universe dates with active exposure (gross > 0 and signal present)."""
    all_dates = {pd.Timestamp(d).normalize() for d in universe_dates}
    if not all_dates or signals.empty:
        return 0.0
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    daily_gross = sig.groupby("date")[gross_col].first()
    daily_active = daily_gross > 1e-12
    active_dates = {d for d, ok in daily_active.items() if ok}
    return len(active_dates & all_dates) / len(all_dates)
