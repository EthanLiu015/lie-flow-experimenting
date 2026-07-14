"""Walk-forward and out-of-regime validation for LieFlow vs raw-vol strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lieflow_quant.backtest import BacktestResult, build_raw_vol_signals, run_backtest, summarize_returns
from lieflow_quant.cache import build_signals_from_cache
from lieflow_quant.panel import (
    DailyCrossSection,
    build_daily_cross_sections,
    compute_forward_returns,
    load_equity_panel,
)


@dataclass(frozen=True)
class EvalPeriod:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


DEFAULT_PERIODS: tuple[EvalPeriod, ...] = (
    EvalPeriod("is_2015_2022", pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31")),
    EvalPeriod("oos_2023_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    EvalPeriod("wf_2015_2017", pd.Timestamp("2015-01-01"), pd.Timestamp("2017-12-31")),
    EvalPeriod("wf_2018_2019", pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31")),
    EvalPeriod("wf_2020_2021", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
    EvalPeriod("wf_2022_2024", pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
)

DEFAULT_COSTS_BPS: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)


def universe_by_date_from_sections(sections: list[DailyCrossSection]) -> dict[pd.Timestamp, list[str]]:
    return {pd.Timestamp(s.date).normalize(): list(s.tickers) for s in sections}


def filter_signals_by_period(signals: pd.DataFrame, period: EvalPeriod) -> pd.DataFrame:
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    mask = (sig["date"] >= period.start) & (sig["date"] <= period.end)
    return sig.loc[mask].copy()


def build_panel_sections(
    close: pd.DataFrame,
    *,
    n_target: int = 50,
    min_stocks: int = 40,
    vix: pd.Series | None = None,
) -> list[DailyCrossSection]:
    return build_daily_cross_sections(
        close,
        n_target=n_target,
        min_stocks=min_stocks,
        vix=vix,
    )


def build_lieflow_signals(
    cache: pd.DataFrame,
    *,
    signal_feature: str = "canonical_vol",
    signal_smoothing: int = 20,
    signal_sign: float = 1.0,
) -> pd.DataFrame:
    return build_signals_from_cache(
        cache,
        signal_feature=signal_feature,
        signal_sign=signal_sign,
        signal_smoothing=signal_smoothing,
    )


def build_raw_vol_signals_for_sections(
    close: pd.DataFrame,
    sections: list[DailyCrossSection],
    *,
    vol_window: int = 20,
    signal_smoothing: int = 20,
    signal_sign: float = 1.0,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex([s.date for s in sections])
    universe = universe_by_date_from_sections(sections)
    return build_raw_vol_signals(
        close,
        dates,
        universe_by_date=universe,
        vol_window=vol_window,
        signal_smoothing=signal_smoothing,
        signal_sign=signal_sign,
    )


def evaluate_signals_in_period(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    period: EvalPeriod,
    *,
    cost_bps: float = 10.0,
    lag: int = 1,
    forward_horizon: int = 1,
    forward_returns: pd.DataFrame | None = None,
) -> BacktestResult | None:
    period_signals = filter_signals_by_period(signals, period)
    if period_signals.empty:
        return None

    fwd = forward_returns
    if fwd is None:
        fwd = compute_forward_returns(close, horizon=forward_horizon)
    result = run_backtest(
        period_signals,
        fwd,
        lag=lag,
        cost_bps=cost_bps,
        forward_horizon=forward_horizon,
        fill_calendar=True,
    )
    return result


def evaluate_multiwindow_fast(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    periods: tuple[EvalPeriod, ...] = DEFAULT_PERIODS,
    *,
    cost_bps: float = 10.0,
    lag: int = 1,
    hold_days: int = 1,
    long_n: int = 0,
    short_n: int = 0,
) -> dict[str, dict]:
    """
    One backtest pass, slice daily returns/IC by period (hold_days=1).

    ~6x faster than calling ``evaluate_signals_in_period`` per window.
    """
    if hold_days > 1:
        raise ValueError("evaluate_multiwindow_fast requires hold_days=1")

    result = run_backtest(
        signals,
        forward_returns,
        lag=lag,
        cost_bps=cost_bps,
        hold_days=hold_days,
        fill_calendar=True,
        long_n=long_n,
        short_n=short_n,
    )
    daily = result.daily_returns
    turnover = result.turnover
    daily_ic = result.daily_ic

    period_metrics: dict[str, dict] = {}
    for period in periods:
        mask = (daily.index >= period.start) & (daily.index <= period.end)
        sub = daily.loc[mask]
        if sub.empty:
            period_metrics[period.name] = {
                "sharpe": float("nan"),
                "mean_ic": float("nan"),
                "total_return": 0.0,
                "n_days": 0,
                "mean_turnover": 0.0,
            }
            continue
        m = summarize_returns(sub)
        if sub.std() < 1e-12:
            m["sharpe"] = float("nan")
            m["total_return"] = 0.0
        if daily_ic is not None:
            ic_aligned = daily_ic.reindex(daily.index)
            ic_sub = ic_aligned.loc[mask]
        else:
            ic_sub = pd.Series(dtype=float)
        period_metrics[period.name] = {
            "sharpe": m["sharpe"],
            "mean_ic": float(ic_sub.mean()) if not ic_sub.empty else float("nan"),
            "total_return": m["total_return"],
            "annualized_return": m["annualized_return"],
            "max_drawdown": m["max_drawdown"],
            "n_days": len(sub),
            "mean_turnover": float(turnover.loc[mask].mean()),
        }
    return period_metrics


def run_validation_grid(
    *,
    lieflow_signals: pd.DataFrame,
    raw_vol_signals: pd.DataFrame,
    close: pd.DataFrame,
    periods: tuple[EvalPeriod, ...] = DEFAULT_PERIODS,
    costs_bps: tuple[float, ...] = DEFAULT_COSTS_BPS,
    universe_label: str = "n50",
    lag: int = 1,
) -> pd.DataFrame:
    rows: list[dict] = []
    for period in periods:
        for cost_bps in costs_bps:
            for strategy, signals in (
                ("lieflow_vol", lieflow_signals),
                ("raw_vol", raw_vol_signals),
            ):
                result = evaluate_signals_in_period(
                    signals,
                    close,
                    period,
                    cost_bps=cost_bps,
                    lag=lag,
                )
                if result is None:
                    continue
                m = result.metrics
                rows.append(
                    {
                        "strategy": strategy,
                        "universe": universe_label,
                        "period": period.name,
                        "period_start": period.start.date().isoformat(),
                        "period_end": period.end.date().isoformat(),
                        "cost_bps": cost_bps,
                        "lag": lag,
                        "sharpe": m["sharpe"],
                        "mean_ic": m["mean_ic"],
                        "total_return": m["total_return"],
                        "annualized_return": m["annualized_return"],
                        "annualized_vol": m["annualized_vol"],
                        "max_drawdown": m["max_drawdown"],
                        "mean_turnover": m["mean_turnover"],
                        "n_days": m["n_days"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_validation(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_rows": 0}

    def _agg(group: pd.DataFrame) -> dict:
        return {
            "n_configs": int(len(group)),
            "pct_sharpe_positive": float((group["sharpe"] > 0).mean()),
            "mean_sharpe": float(group["sharpe"].mean()),
            "min_sharpe": float(group["sharpe"].min()),
            "max_sharpe": float(group["sharpe"].max()),
        }

    summary: dict = {"n_rows": int(len(df))}
    for strategy in sorted(df["strategy"].unique()):
        s = df[df["strategy"] == strategy]
        summary[strategy] = _agg(s)
        summary[f"{strategy}_by_period"] = {
            period: _agg(s[s["period"] == period])
            for period in sorted(s["period"].unique())
        }
        summary[f"{strategy}_by_universe"] = {
            uni: _agg(s[s["universe"] == uni])
            for uni in sorted(s["universe"].unique())
        }
        oos_rows = []
        for uni in sorted(s["universe"].unique()):
            oos = s[(s["period"] == "oos_2023_2024") & (s["cost_bps"] == 10.0) & (s["universe"] == uni)]
            if not oos.empty:
                oos_rows.append(oos.iloc[0].to_dict())
        summary[f"{strategy}_oos_at_10bps"] = oos_rows
    return summary


def save_validation_outputs(
    df: pd.DataFrame,
    summary: dict,
    output_dir: Path | str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "validation_grid.csv", index=False)
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
