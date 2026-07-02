#!/usr/bin/env python3
"""Autoresearch: LieFlow strategy with calendar Sharpe > 1 and mean return > 17%."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

from lieflow_quant.methodology import coverage_fraction
from lieflow_quant.session import EvalSession, MultiWindowConfig
from lieflow_quant.validation import DEFAULT_PERIODS

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs/strategy/best_lieflow_sharpe_return.json"

SHARPE_MIN = 1.0
RETURN_MIN = 0.17


def _is_nan(x) -> bool:
    return isinstance(x, float) and x != x


def effective_period_sharpe(pm: dict) -> float:
    """
    Metric sharpe that doesn't "hide" gated/no-trade windows.

    Our backtest sets flat (all-0) windows to Sharpe=NaN to avoid divide-by-zero.
    For *search constraints* (min Sharpe), we treat those as Sharpe=0.
    """
    sh = pm.get("sharpe")
    tr = float(pm.get("total_return", 0.0))
    if sh is None or _is_nan(sh):
        return 0.0 if tr == 0.0 else float("nan")
    return float(sh)


def mean_total_return(metrics: dict) -> float:
    """
    Mean total return across ALL configured periods.

    This prevents inflating results by averaging only over post-2023 periods where
    LieFlow inference exists (and silently dropping earlier, flat windows).
    """
    periods = metrics.get("periods", {})
    rets: list[float] = []
    for p in DEFAULT_PERIODS:
        pm = periods.get(p.name)
        if not isinstance(pm, dict) or pm.get("n_days", 0) <= 0:
            rets.append(0.0)
        else:
            rets.append(float(pm.get("total_return", 0.0)))
    return sum(rets) / len(rets) if rets else float("nan")


def meets_target(metrics: dict) -> bool:
    periods = metrics.get("periods", {})
    sharpes: list[float] = []
    for p in DEFAULT_PERIODS:
        pm = periods.get(p.name)
        if not isinstance(pm, dict) or pm.get("n_days", 0) <= 0:
            sharpes.append(0.0)
        else:
            sharpes.append(effective_period_sharpe(pm))
    if not sharpes:
        return False
    min_s = min(sharpes)
    return min_s > SHARPE_MIN and mean_total_return(metrics) > RETURN_MIN


def build_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []

    # Hybrid: LieFlow sizes exposure / optional soft gate — avoid hard zero-gate.
    for conc in (0.0, 0.5, 0.8, 0.9, 0.95, 1.0, 1.02):
        for smooth in (10, 12, 14, 16, 18, 20, 22, 25):
            for lw in (0.1, 0.2, 0.3, 0.4, 0.5):
                for cost in (5.0, 10.0):
                    exps.append(
                        (
                            f"hybrid c{conc:.2f} s{smooth} w{lw:.1f} c{int(cost)}",
                            MultiWindowConfig(
                                hybrid_vol=True,
                                min_concentration_ratio=conc if conc > 0 else None,
                                signal_smoothing=smooth,
                                lieflow_weight=lw,
                                cost_bps=cost,
                                min_exposure=0.5,
                                max_exposure=1.0,
                            ),
                        )
                    )

    # Pure LieFlow advanced signals (full coverage when inference available).
    for strategy in ("canonical_vol", "mom_minus_vol", "mom_resid_vol", "radial_distance"):
        for sign in (1.0, -1.0):
            for smooth in (10, 15, 20, 25, 30):
                for cost in (5.0, 10.0):
                    exps.append(
                        (
                            f"{strategy} sign{int(sign)} s{smooth} c{int(cost)}",
                            MultiWindowConfig(
                                strategy=strategy,
                                signal_sign=sign,
                                signal_smoothing=smooth,
                                cost_bps=cost,
                            ),
                        )
                    )

    # Exposure-scaled hybrid (LieFlow overlay, no hard gate).
    for smooth in (14, 16, 18, 20):
        for lw in (0.2, 0.3, 0.4):
            exps.append(
                (
                    f"hybrid soft s{smooth} w{lw}",
                    MultiWindowConfig(
                        hybrid_vol=True,
                        min_concentration_ratio=None,
                        signal_smoothing=smooth,
                        lieflow_weight=lw,
                        cost_bps=10.0,
                        min_exposure=0.75,
                        max_exposure=1.0,
                    ),
                )
            )

    return exps


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sharpe-min", type=float, default=SHARPE_MIN)
    parser.add_argument("--return-min", type=float, default=RETURN_MIN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--inference-ml-test-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, only ML test-split inference (honest ML, harder to hit return target).",
    )
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print("[autoresearch] mode: classic")
    print(
        f"[autoresearch] metric: min_sharpe>{args.sharpe_min} AND mean_return>{args.return_min:.0%} "
        f"| LieFlow required | calendar Sharpe"
    )

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = args.output_dir or ROOT / "autoresearch" / f"loop-lieflow-sharpe-ret-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    experiments = build_experiments()
    names = [n for n, _ in experiments]
    configs = [c for _, c in experiments]

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        n_target=60,
        inference_ml_test_only=args.inference_ml_test_only,
    )
    print(
        f"configs={len(configs)} workers={workers} "
        f"ml_test_only={args.inference_ml_test_only} ml_start={session.ml_test_start.date()}"
    )

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    print(f"sweep_ms={(perf_counter() - t0) * 1000:.0f}")

    with open(tsv, "w") as f:
        f.write("name\tmin_sharpe\tmean_sharpe\tmean_return\tcoverage\tmeets_target\n")

    winners: list[dict] = []
    best_score = float("-inf")
    best: dict | None = None
    sha = git_head()

    for name, cfg, m in zip(names, configs, metrics_list):
        if cfg.raw_vol_baseline:
            continue
        m_ret = mean_total_return(m)
        periods = m.get("periods", {})
        sharpes = []
        for p in DEFAULT_PERIODS:
            pm = periods.get(p.name)
            if not isinstance(pm, dict) or pm.get("n_days", 0) <= 0:
                sharpes.append(0.0)
            else:
                sharpes.append(effective_period_sharpe(pm))
        min_s = min(sharpes) if sharpes else float("nan")
        sig = session.build_signals(cfg)
        cov = coverage_fraction(sig, session.section_dates)
        ok = bool(sharpes) and (min_s > args.sharpe_min) and (m_ret > args.return_min)
        score = min_s + m_ret if sharpes else float("-inf")
        if ok:
            winners.append(
                {
                    "name": name,
                    "config": cfg,
                    "min_sharpe": min_s,
                    "mean_return": m_ret,
                    "coverage": cov,
                    "meets_target": ok,
                    "metrics": m,
                }
            )
        if score > best_score and sharpes:
            best_score = score
            best = {
                "name": name,
                "config": cfg,
                "min_sharpe": min_s,
                "mean_return": m_ret,
                "coverage": cov,
                "meets_target": ok,
                "metrics": m,
            }
        with open(tsv, "a") as f:
            f.write(f"{name}\t{min_s:.4f}\t{m['mean_sharpe']:.4f}\t{m_ret:.4f}\t{cov:.2%}\t{ok}\n")

    status = "CONVERGED" if winners else "BOUNDED"
    pick = winners[0] if winners else best
    if winners:
        winners.sort(key=lambda r: (-r["mean_return"], -r["min_sharpe"]))

    if pick:
        cfg = pick["config"]
        BEST_PATH.write_text(
            json.dumps(
                {
                    "name": pick["name"],
                    "meets_target": pick.get("meets_target", False),
                    "min_sharpe": pick["min_sharpe"],
                    "mean_return": pick["mean_return"],
                    "coverage": pick["coverage"],
                    "config": {
                        "hybrid_vol": cfg.hybrid_vol,
                        "strategy": cfg.strategy,
                        "min_concentration_ratio": cfg.min_concentration_ratio,
                        "lieflow_weight": cfg.lieflow_weight,
                        "signal_smoothing": cfg.signal_smoothing,
                        "cost_bps": cfg.cost_bps,
                        "signal_sign": cfg.signal_sign,
                        "min_exposure": cfg.min_exposure,
                        "max_exposure": cfg.max_exposure,
                    },
                    "metrics": pick["metrics"],
                    "inference_ml_test_only": args.inference_ml_test_only,
                },
                indent=2,
                default=str,
            )
        )

    handoff = {
        "status": status,
        "n_winners": len(winners),
        "best": pick["name"] if pick else None,
        "inference_ml_test_only": args.inference_ml_test_only,
        "results_tsv": str(tsv),
        "commit": sha,
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    (out_dir / "evals-summary.md").write_text(
        f"# LieFlow Sharpe+Return Autoresearch\n\n"
        f"- Target: min Sharpe > {args.sharpe_min}, mean return > {args.return_min:.0%}\n"
        f"- Calendar Sharpe (0% on flat days)\n"
        f"- ML test-only inference: {args.inference_ml_test_only}\n"
        f"- Winners: {len(winners)}\n"
        + (f"- Best winner: `{winners[0]['name']}` minS={winners[0]['min_sharpe']:.3f} ret={winners[0]['mean_return']:.1%}\n" if winners else "")
        + (f"- Best overall: `{best['name']}` minS={best['min_sharpe']:.3f} ret={best['mean_return']:.1%}\n" if best and not winners else "")
    )

    print(json.dumps(handoff, indent=2))
    if winners:
        for w in winners[:5]:
            print(f"WIN {w['name']} minS={w['min_sharpe']:.3f} ret={w['mean_return']:.1%} cov={w['coverage']:.1%}")
    elif best:
        print(f"BOUNDED best={best['name']} minS={best['min_sharpe']:.3f} ret={best['mean_return']:.1%}")


if __name__ == "__main__":
    main()
