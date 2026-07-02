#!/usr/bin/env python3
"""Audit signal date → trade date → forward return alignment and lookahead."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lieflow_quant.backtest import (
    _cross_sectional_weights_np,
    _spearman_ic,
    _turnover_dict,
    run_backtest,
    summarize_returns,
)
from lieflow_quant.panel import compute_forward_returns, load_equity_panel
from lieflow_quant.session import EvalSession, MultiWindowConfig

ROOT = Path(__file__).resolve().parents[1]


def manual_day_return(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    lag: int = 1,
    cost_bps: float = 10.0,
    prev_weights: dict[str, float] | None = None,
) -> dict:
    """Recompute one day exactly as run_backtest should."""
    bday = pd.tseries.offsets.BDay(lag)
    trade_date = pd.Timestamp(signal_date) + bday

    day = signals[signals["date"] == signal_date]
    if day.empty:
        return {"error": "no signals", "signal_date": signal_date, "trade_date": trade_date}

    gross = float(day["gross_exposure"].iloc[0])
    tickers = day["ticker"].astype(str).to_numpy()
    sig_vals = day["signal"].to_numpy(dtype=float)

    if trade_date not in forward_returns.index:
        return {"error": "trade_date missing in fwd", "trade_date": trade_date}

    if gross < 1e-12:
        w_dict: dict[str, float] = {}
        turnover = _turnover_dict(w_dict, prev_weights)
        net_ret = -turnover * cost_bps / 10_000
        return {
            "signal_date": signal_date,
            "trade_date": trade_date,
            "gross": gross,
            "port_ret": 0.0,
            "turnover": turnover,
            "net_ret": net_ret,
            "n_names": 0,
        }

    target_w = _cross_sectional_weights_np(sig_vals, gross)
    fwd_row = forward_returns.loc[trade_date]
    col_pos = {str(c): i for i, c in enumerate(forward_returns.columns)}

    valid_mask = []
    w_list = []
    f_list = []
    for t, w in zip(tickers, target_w, strict=True):
        if t in col_pos and np.isfinite(w) and np.isfinite(fwd_row[t]):
            valid_mask.append(t)
            w_list.append(w)
            f_list.append(float(fwd_row[t]))

    if len(valid_mask) < 5:
        return {"error": "too few names", "n": len(valid_mask)}

    w_arr = np.array(w_list)
    f_arr = np.array(f_list)
    gross_w = float(np.abs(w_arr).sum())
    w_arr = w_arr / gross_w * gross
    port_ret = float((w_arr * f_arr).sum())
    w_dict = dict(zip(valid_mask, w_arr, strict=True))
    turnover = _turnover_dict(w_dict, prev_weights)
    net_ret = port_ret - turnover * cost_bps / 10_000
    ic = _spearman_ic(sig_vals[: len(valid_mask)], f_arr)  # approximate

    fwd_at_signal = forward_returns.loc[signal_date] if signal_date in forward_returns.index else None
    wrong_ret = None
    if fwd_at_signal is not None:
        w_wrong = _cross_sectional_weights_np(sig_vals, gross)
        wrong_parts = []
        for t, w in zip(tickers, w_wrong, strict=True):
            if t in col_pos and np.isfinite(w) and np.isfinite(fwd_at_signal[t]):
                wrong_parts.append(w * float(fwd_at_signal[t]))
        if len(wrong_parts) >= 5:
            wrong_ret = float(np.sum(wrong_parts))

    return {
        "signal_date": signal_date,
        "trade_date": trade_date,
        "fwd_used": f"close[{trade_date}] -> close[{trade_date + pd.tseries.offsets.BDay(1)}]",
        "gross": gross,
        "port_ret": port_ret,
        "turnover": turnover,
        "net_ret": net_ret,
        "n_names": len(valid_mask),
        "ic": ic,
        "lookahead_if_used_signal_date_fwd": wrong_ret,
    }


def lag_sensitivity(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    cost_bps: float,
    lags: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    rows = []
    for lag in lags:
        r = run_backtest(signals, forward_returns, lag=lag, cost_bps=cost_bps, fill_calendar=True)
        m = r.metrics
        rows.append(
            {
                "lag": lag,
                "total_return": m["total_return"],
                "sharpe": m["sharpe"],
                "n_trade_days": m["n_trade_days"],
                "mean_ic": m["mean_ic"],
            }
        )
    return pd.DataFrame(rows)


def shuffle_ic_test(signals: pd.DataFrame, forward_returns: pd.DataFrame, *, lag: int = 1, n: int = 20) -> dict:
    """If IC stays high after shuffling signal dates, likely misalignment."""
    base = run_backtest(signals, forward_returns, lag=lag, fill_calendar=False)
    base_ic = float(base.metrics["mean_ic"]) if base.daily_ic is not None else float("nan")

    rng = np.random.default_rng(42)
    dates = signals["date"].unique()
    shuffled_ics = []
    for _ in range(n):
        perm = rng.permutation(dates)
        shuf = signals.copy()
        date_map = dict(zip(dates, perm, strict=True))
        shuf["date"] = shuf["date"].map(date_map)
        r = run_backtest(shuf, forward_returns, lag=lag, fill_calendar=False)
        if r.daily_ic is not None and not r.daily_ic.empty:
            shuffled_ics.append(float(r.daily_ic.mean()))

    return {
        "base_ic": base_ic,
        "shuffled_ic_mean": float(np.mean(shuffled_ics)) if shuffled_ics else float("nan"),
        "shuffled_ic_std": float(np.std(shuffled_ics)) if shuffled_ics else float("nan"),
    }


def compare_period_returns(session: EvalSession, config: MultiWindowConfig) -> dict:
    signals = session.build_signals(config)
    oos_start = pd.Timestamp("2023-01-01")
    oos_end = pd.Timestamp("2024-12-31")

    sig_oos = signals[(signals["date"] >= oos_start) & (signals["date"] <= oos_end)]
    full = run_backtest(signals, session.forward_returns, lag=config.lag, cost_bps=config.cost_bps, fill_calendar=True)
    oos_only = run_backtest(
        sig_oos, session.forward_returns, lag=config.lag, cost_bps=config.cost_bps, fill_calendar=True
    )

    # Period slice from full run
    mask = (full.daily_returns.index >= oos_start) & (full.daily_returns.index <= oos_end)
    sliced = full.daily_returns.loc[mask]
    sliced_m = summarize_returns(sliced)

    return {
        "full_total_return": full.metrics["total_return"],
        "oos_backtest_only_total": oos_only.metrics["total_return"],
        "oos_sliced_from_full_total": sliced_m["total_return"],
        "oos_sliced_sharpe": sliced_m["sharpe"],
        "full_sharpe": full.metrics["sharpe"],
        "oos_trade_days": oos_only.metrics["n_trade_days"],
        "oos_mean_daily_when_active": float(
            oos_only.daily_returns[oos_only.gross_exposure > 1e-12].mean()
        )
        if (oos_only.gross_exposure > 1e-12).any()
        else 0.0,
    }


def main() -> None:
    best_path = ROOT / "outputs/strategy/best_lieflow_sharpe_return.json"
    if best_path.exists():
        saved = json.loads(best_path.read_text())
        cfg_dict = saved["config"]
        config = MultiWindowConfig(
            hybrid_vol=cfg_dict["hybrid_vol"],
            min_concentration_ratio=cfg_dict.get("min_concentration_ratio"),
            lieflow_weight=cfg_dict.get("lieflow_weight", 0.5),
            signal_smoothing=cfg_dict["signal_smoothing"],
            cost_bps=cfg_dict["cost_bps"],
            min_exposure=cfg_dict.get("min_exposure", 0.5),
            max_exposure=cfg_dict.get("max_exposure", 1.0),
        )
        ml_test_only = saved.get("inference_ml_test_only", True)
    else:
        config = MultiWindowConfig(
            hybrid_vol=True,
            min_concentration_ratio=0.5,
            lieflow_weight=0.1,
            signal_smoothing=18,
            cost_bps=5.0,
        )
        ml_test_only = True

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        n_target=60,
        inference_ml_test_only=ml_test_only,
    )
    signals = session.build_signals(config)
    fwd = session.forward_returns

    print("=== Config ===")
    print(config)
    print(f"ml_test_only={ml_test_only} cache_rows={len(session.cache)}")
    print(f"signal_dates={signals['date'].nunique()} active={(signals.groupby('date')['gross_exposure'].first() > 1e-12).sum()}")

    # 1. Forward return definition check
    close, _ = load_equity_panel(ROOT / "data/equity")
    px = close[[c for c in close.columns if c != "^VIX"]]
    ticker = px.columns[0]
    sample_dates = px.index[100:105]
    print("\n=== Forward return definition (one ticker) ===")
    for d in sample_dates:
        d1 = d + pd.tseries.offsets.BDay(1)
        if d1 not in px.index:
            continue
        manual = float(px.loc[d1, ticker] / px.loc[d, ticker] - 1)
        cached = float(fwd.loc[d, ticker]) if d in fwd.index else float("nan")
        print(f"  fwd[{d.date()}] manual ret {d.date()}->{d1.date()} = {manual:.6f}  cached={cached:.6f}")

    # 2. Manual vs backtest on sample active days
    active_dates = (
        signals.groupby("date")["gross_exposure"]
        .first()
        .loc[lambda s: s > 1e-12]
        .index.sort_values()
    )
    sample_active = list(active_dates[50:55])
    print("\n=== Manual vs backtest (5 active OOS days) ===")
    bt = run_backtest(signals, fwd, lag=1, cost_bps=config.cost_bps, fill_calendar=False)
    prev_w = None
    for sd in sample_active:
        manual = manual_day_return(signals, fwd, sd, lag=1, cost_bps=config.cost_bps, prev_weights=prev_w)
        trade_date = manual["trade_date"]
        bt_ret = float(bt.daily_returns.get(trade_date, float("nan")))
        match = abs(manual["net_ret"] - bt_ret) < 1e-10 if not np.isnan(bt_ret) else False
        print(
            f"  signal={sd.date()} trade={trade_date.date()} "
            f"manual={manual['net_ret']:.6f} bt={bt_ret:.6f} match={match} "
            f"lookahead_same_day_fwd={manual.get('lookahead_if_used_signal_date_fwd')}"
        )

    # 3. Lag sensitivity
    print("\n=== Lag sensitivity (full sample) ===")
    print(lag_sensitivity(signals, fwd, cost_bps=config.cost_bps).to_string(index=False))

    # 4. Shuffle IC test
    oos_sig = signals[signals["date"] >= pd.Timestamp("2023-01-01")]
    print("\n=== Shuffle-date IC test (OOS signals) ===")
    print(shuffle_ic_test(oos_sig, fwd, lag=1))

    # 5. Cost sensitivity
    print("\n=== Cost sensitivity ===")
    for cost in (0.0, 5.0, 10.0, 20.0):
        r = run_backtest(oos_sig, fwd, lag=1, cost_bps=cost, fill_calendar=True)
        m = summarize_returns(r.daily_returns)
        print(f"  cost={cost:4.0f}bps total_ret={m['total_return']:.2%} sharpe={m['sharpe']:.2f}")

    # 6. Period return consistency
    print("\n=== OOS return consistency ===")
    print(compare_period_returns(session, config))

    # 7. Raw vol baseline comparison (no LieFlow gate)
    raw_cfg = MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=config.signal_smoothing, cost_bps=10.0)
    raw_sig = session.build_signals(raw_cfg)
    raw_oos = raw_sig[raw_sig["date"] >= pd.Timestamp("2023-01-01")]
    raw_r = run_backtest(raw_oos, fwd, lag=1, cost_bps=10.0, fill_calendar=True)
    hybrid_r = run_backtest(oos_sig, fwd, lag=1, cost_bps=10.0, fill_calendar=True)
    print("\n=== OOS 2023+ @ 10bps: hybrid vs raw vol ===")
    print(f"  hybrid total_ret={hybrid_r.metrics['total_return']:.2%} sharpe={hybrid_r.metrics['sharpe']:.2f}")
    print(f"  raw_vol total_ret={raw_r.metrics['total_return']:.2%} sharpe={raw_r.metrics['sharpe']:.2f}")

    out = ROOT / "outputs/strategy/backtest_audit.json"
    audit = {
        "config": str(config),
        "ml_test_only": ml_test_only,
        "lag_sensitivity": lag_sensitivity(signals, fwd, cost_bps=config.cost_bps).to_dict(orient="records"),
        "shuffle_ic": shuffle_ic_test(oos_sig, fwd),
        "oos_consistency": compare_period_returns(session, config),
    }
    out.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
