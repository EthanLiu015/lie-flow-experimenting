#!/usr/bin/env python3
"""
Autoresearch: VIX × concentration regime switcher.

Optimizes holdout (oos_2023_2024) composite:
  portfolio Sharpe + annualized return + drawdown bonus (-max_drawdown is positive)

Uses pre-2023-trained LieFlow cache; pre-2023 dates fall back to raw vol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

from lieflow_quant.backtest import run_backtest
from lieflow_quant.methodology import (
    effective_period_sharpe,
    lieflow_alpha_influence_fraction,
    period_annualized_return,
)
from lieflow_quant.session import EvalSession, MultiWindowConfig
from lieflow_quant.validation import (
    DEFAULT_PERIODS,
    build_raw_vol_signals_for_sections,
    filter_signals_by_period,
)

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs/strategy/best_regime_switcher.json"
CACHE_PATH = ROOT / "outputs/strategy/inference_cache_pre2023train_oos2023.npz"
HOLDOUT = "oos_2023_2024"


def build_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []
    base = dict(cost_bps=10.0, hybrid_fallback=True, hybrid_regime_switcher=True)

    strategies = ("mom_resid_vol", "mom_minus_vol")
    smoothings = (16, 18, 20)
    lieflow_weights = (0.25, 0.4, 0.55)
    conc_pairs = ((1.0, 0.85), (1.05, 0.8), (0.95, 0.75))
    min_exp = (0.25, 0.35)
    neutral = (0.55, 0.7)
    boost_sets = (("mid", "low"), ("mid",))

    for strat, smooth, lw, (cg, cb), mn, neu, boost in itertools.product(
        strategies,
        smoothings,
        lieflow_weights,
        conc_pairs,
        min_exp,
        neutral,
        boost_sets,
    ):
        tag = (
            f"regime {strat} sm{smooth} w{lw:.2f} cg{cg:.2f} cb{cb:.2f} "
            f"mn{mn:.2f} neu{neu:.2f} boost{'+'.join(boost)}"
        )
        exps.append(
            (
                tag,
                MultiWindowConfig(
                    strategy=strat,
                    signal_smoothing=smooth,
                    lieflow_weight=lw,
                    conc_good_ratio=cg,
                    conc_bad_ratio=cb,
                    min_exposure=mn,
                    max_exposure=1.0,
                    exposure_neutral=neu,
                    vix_boost_regimes=boost,
                    vix_cut_regimes=("high",),
                    **base,
                ),
            )
        )
    return exps


def holdout_period(metrics: dict) -> dict:
    return metrics["periods"][HOLDOUT]


def composite_score(pm: dict) -> float:
    sharpe = effective_period_sharpe(pm)
    ann = period_annualized_return(pm)
    max_dd = float(pm.get("max_drawdown", 0.0))
    return sharpe + ann + 0.75 * (-max_dd)


def recruiter_metrics(session: EvalSession, cfg: MultiWindowConfig) -> dict:
    period = next(p for p in DEFAULT_PERIODS if p.name == HOLDOUT)
    sig = filter_signals_by_period(session.build_signals(cfg), period)
    bt = run_backtest(sig, session.forward_returns, lag=cfg.lag, cost_bps=cfg.cost_bps)
    daily = bt.metrics
    exposure = float(daily.get("mean_gross_exposure", 0.0))
    port_sharpe = effective_period_sharpe(daily)
    ann_ret = period_annualized_return(daily)
    max_dd = float(daily.get("max_drawdown", 0.0))
    active_rets = bt.daily_returns[bt.gross_exposure > 1e-12]
    trade_sharpe = float("nan")
    if len(active_rets) > 5 and active_rets.std() > 1e-12:
        trade_sharpe = float(active_rets.mean() / active_rets.std() * (252**0.5))
    return {
        "portfolio_sharpe": port_sharpe,
        "trade_sharpe": trade_sharpe,
        "exposure": exposure,
        "annual_return": ann_ret,
        "max_drawdown": max_dd,
        "total_return": float(daily.get("total_return", 0.0)),
        "n_days": int(daily.get("n_days", 0)),
        "n_trade_days": int(daily.get("n_trade_days", 0)),
    }


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--max-configs", type=int, default=0, help="0 = all")
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print("[autoresearch] mode: classic")
    print("[autoresearch] metric: holdout sharpe + ann_return + drawdown | VIX×conc regime switcher")

    if not args.cache.exists():
        raise SystemExit(f"Missing cache {args.cache}")

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = ROOT / "autoresearch" / f"loop-regime-switcher-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    experiments = build_experiments()
    if args.max_configs > 0:
        experiments = experiments[: args.max_configs]
    names = [n for n, _ in experiments]
    configs = [c for _, c in experiments]

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=args.cache,
        n_target=60,
        inference_ml_test_only=False,
    )
    raw_vol = build_raw_vol_signals_for_sections(session.close, session.sections, signal_smoothing=1)
    lieflow_dates = frozenset(session.cache["date"].dt.normalize())

    raw_m = session.evaluate_multiwindow(
        MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=18, cost_bps=10.0)
    )
    raw_hold = holdout_period(raw_m)
    raw_score = composite_score(raw_hold)
    print(
        f"configs={len(configs)} raw_holdout S={effective_period_sharpe(raw_hold):.3f} "
        f"ann={period_annualized_return(raw_hold):.2%} dd={raw_hold['max_drawdown']:.2%} "
        f"score={raw_score:.4f}"
    )

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    print(f"sweep_ms={(perf_counter() - t0) * 1000:.0f}")

    with open(tsv, "w") as f:
        f.write(
            "name\tholdout_sharpe\tholdout_ann\tholdout_dd\tcomposite\talpha_inf\tbeats_raw\n"
        )

    best: dict | None = None
    best_score = float("-inf")

    for name, cfg, m in zip(names, configs, metrics_list):
        hold = holdout_period(m)
        score = composite_score(hold)
        sig = session.build_signals(cfg)
        influence = lieflow_alpha_influence_fraction(sig, raw_vol, lieflow_dates)
        beat = (
            effective_period_sharpe(hold) >= effective_period_sharpe(raw_hold)
            and period_annualized_return(hold) >= period_annualized_return(raw_hold)
            and hold["max_drawdown"] >= raw_hold["max_drawdown"]
        )
        row = {
            "name": name,
            "config": cfg,
            "metrics": m,
            "holdout": hold,
            "composite": score,
            "influence": influence,
            "beats_raw": beat,
        }
        if score > best_score:
            best_score = score
            best = row

        with open(tsv, "a") as f:
            f.write(
                f"{name}\t{effective_period_sharpe(hold):.4f}\t"
                f"{period_annualized_return(hold):.4f}\t{hold['max_drawdown']:.4f}\t"
                f"{score:.4f}\t{influence:.4f}\t{beat}\n"
            )

    if best:
        rec = recruiter_metrics(session, best["config"])
        BEST_PATH.write_text(
            json.dumps(
                {
                    "name": best["name"],
                    "composite_score": best["composite"],
                    "beats_raw_vol_holdout": best["beats_raw"],
                    "influence": best["influence"],
                    "config": {
                        "hybrid_regime_switcher": True,
                        "strategy": best["config"].strategy,
                        "lieflow_weight": best["config"].lieflow_weight,
                        "signal_smoothing": best["config"].signal_smoothing,
                        "conc_good_ratio": best["config"].conc_good_ratio,
                        "conc_bad_ratio": best["config"].conc_bad_ratio,
                        "min_exposure": best["config"].min_exposure,
                        "max_exposure": best["config"].max_exposure,
                        "exposure_neutral": best["config"].exposure_neutral,
                        "vix_boost_regimes": best["config"].vix_boost_regimes,
                        "vix_cut_regimes": best["config"].vix_cut_regimes,
                    },
                    "holdout_recruiter_metrics": rec,
                    "holdout_period_metrics": best["holdout"],
                    "raw_vol_baseline_holdout": raw_hold,
                },
                indent=2,
                default=str,
            )
        )

    handoff = {
        "status": "CONVERGED" if best else "BOUNDED",
        "best": best["name"] if best else None,
        "best_composite": best_score if best else None,
        "results_tsv": str(tsv),
        "commit": git_head(),
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    print(json.dumps(handoff, indent=2))
    if best:
        h = best["holdout"]
        rec = recruiter_metrics(session, best["config"])
        print(
            f"BEST {best['name']} composite={best['composite']:.4f} beats_raw={best['beats_raw']}\n"
            f"  portfolio_sharpe={rec['portfolio_sharpe']:.3f} trade_sharpe={rec['trade_sharpe']:.3f} "
            f"exposure={rec['exposure']:.1%} ann={rec['annual_return']:.2%} max_dd={rec['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
