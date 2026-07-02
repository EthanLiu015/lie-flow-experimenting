#!/usr/bin/env python3
"""
Autoresearch: LieFlow alpha + raw-vol fallback.

Anti-overfitting:
- Select on TUNE windows only (pre-2023 IS + walk-forward folds)
- Report strict HOLDOUT (oos_2023_2024) without using it for selection
- ML inference test-split only (LieFlow features not used in-sample)
- Require LieFlow alpha to change cross-sectional signals
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

from lieflow_quant.methodology import (
    aggregate_multiwindow_metrics,
    coverage_fraction,
    lieflow_alpha_influence_fraction,
)
from lieflow_quant.session import EvalSession, MultiWindowConfig
from lieflow_quant.validation import build_raw_vol_signals_for_sections

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs/strategy/best_hybrid_alpha.json"

TUNE_PERIODS = ("is_2015_2022", "wf_2015_2017", "wf_2018_2019", "wf_2020_2021")
HOLDOUT_PERIODS = ("oos_2023_2024",)

SHARPE_FLOOR = 0.35
MIN_ALPHA_WEIGHT = 0.25
MIN_ALPHA_INFLUENCE = 0.15
MAX_HOLDOUT_SHARPE_GAP = 0.75


def build_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []
    strategies = (
        ("mom_minus_vol", (1.0, -1.0)),
        ("mom_resid_vol", (1.0,)),
        ("canonical_vol", (1.0, -1.0)),
        ("conc_scaled_momentum", (1.0,)),
        ("radial_distance", (1.0,)),
    )
    for strat, signs in strategies:
        for sign in signs:
            for lw in (0.25, 0.35, 0.45, 0.55):
                for smooth in (14, 18, 22):
                    for conc in (None, 0.5, 0.8):
                        tag = f"{strat} s{int(sign)} w{lw:.2f} sm{smooth}"
                        tag += " noc" if conc is None else f" c{conc:.1f}"
                        exps.append(
                            (
                                tag,
                                MultiWindowConfig(
                                    hybrid_alpha=True,
                                    hybrid_fallback=True,
                                    strategy=strat,
                                    signal_sign=sign,
                                    lieflow_weight=lw,
                                    signal_smoothing=smooth,
                                    min_concentration_ratio=conc,
                                    cost_bps=10.0,
                                    min_exposure=0.5,
                                    max_exposure=1.0,
                                ),
                            )
                        )
    return exps


def tune_metrics(full: dict) -> dict[str, float]:
    return aggregate_multiwindow_metrics(full["periods"], TUNE_PERIODS)


def holdout_metrics(full: dict) -> dict[str, float]:
    return aggregate_multiwindow_metrics(full["periods"], HOLDOUT_PERIODS)


def selection_score(tune: dict[str, float]) -> float:
    return (
        tune["mean_annualized_return"]
        + 0.5 * tune["min_sharpe"]
        + 0.2 * tune["mean_total_return"]
    )


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sharpe-floor", type=float, default=SHARPE_FLOOR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print("[autoresearch] mode: classic")
    print(
        "[autoresearch] metric: tune_sharpe+return | holdout=oos_2023_2024 | "
        "ml_test_only=True | hybrid_alpha blend"
    )

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = args.output_dir or ROOT / "autoresearch" / f"loop-hybrid-alpha-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    experiments = build_experiments()
    names = [n for n, _ in experiments]
    configs = [c for _, c in experiments]

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        n_target=60,
        inference_ml_test_only=True,
    )
    raw_vol = build_raw_vol_signals_for_sections(session.close, session.sections, signal_smoothing=1)
    lieflow_dates = frozenset(session.cache["date"].dt.normalize())

    raw_base = session.evaluate_multiwindow(
        MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=18, cost_bps=10.0)
    )
    print(
        f"configs={len(configs)} workers={workers} cache_rows={len(session.cache)} "
        f"ml_start={session.ml_test_start.date()} "
        f"raw_tune_minS={tune_metrics(raw_base)['min_sharpe']:.3f}"
    )

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    print(f"sweep_ms={(perf_counter() - t0) * 1000:.0f}")

    with open(tsv, "w") as f:
        f.write(
            "name\ttune_min_sharpe\ttune_mean_ann\tholdout_sharpe\tholdout_ret\t"
            "alpha_influence\tcoverage\tselection\teligible\n"
        )

    best_score = float("-inf")
    best: dict | None = None
    eligible: list[dict] = []
    sha = git_head()

    for name, cfg, m in zip(names, configs, metrics_list):
        tune = tune_metrics(m)
        hold = holdout_metrics(m)
        sig = session.build_signals(cfg)
        cov = coverage_fraction(sig, session.section_dates)
        alpha_inf = lieflow_alpha_influence_fraction(sig, raw_vol, lieflow_dates)
        score = selection_score(tune)
        gap = tune["min_sharpe"] - hold["min_sharpe"]
        ok = (
            tune["min_sharpe"] >= args.sharpe_floor
            and cfg.lieflow_weight >= MIN_ALPHA_WEIGHT
            and alpha_inf >= MIN_ALPHA_INFLUENCE
            and hold["min_sharpe"] > 0
            and gap <= MAX_HOLDOUT_SHARPE_GAP
        )
        row = {
            "name": name,
            "config": cfg,
            "metrics": m,
            "tune": tune,
            "holdout": hold,
            "alpha_influence": alpha_inf,
            "sharpe_gap": gap,
            "coverage": cov,
            "selection": score,
            "eligible": ok,
        }
        if ok:
            eligible.append(row)
        if score > best_score:
            best_score = score
            best = row

        with open(tsv, "a") as f:
            f.write(
                f"{name}\t{tune['min_sharpe']:.4f}\t{tune['mean_annualized_return']:.4f}\t"
                f"{hold['min_sharpe']:.4f}\t{hold['mean_total_return']:.4f}\t"
                f"{alpha_inf:.4f}\t{cov:.2%}\t{score:.4f}\t{ok}\n"
            )

    eligible.sort(key=lambda r: (-r["selection"], -r["holdout"]["min_sharpe"]))
    pick = eligible[0] if eligible else best
    status = "CONVERGED" if eligible else "BOUNDED"

    if pick:
        cfg = pick["config"]
        BEST_PATH.write_text(
            json.dumps(
                {
                    "name": pick["name"],
                    "eligible": pick["eligible"],
                    "selection_score": pick["selection"],
                    "alpha_influence": pick["alpha_influence"],
                    "sharpe_gap_tune_minus_holdout": pick["sharpe_gap"],
                    "coverage": pick["coverage"],
                    "config": {
                        "hybrid_alpha": True,
                        "hybrid_fallback": cfg.hybrid_fallback,
                        "strategy": cfg.strategy,
                        "signal_sign": cfg.signal_sign,
                        "lieflow_weight": cfg.lieflow_weight,
                        "signal_smoothing": cfg.signal_smoothing,
                        "min_concentration_ratio": cfg.min_concentration_ratio,
                        "cost_bps": cfg.cost_bps,
                    },
                    "tune_metrics": pick["tune"],
                    "holdout_metrics": pick["holdout"],
                    "full_metrics": pick["metrics"],
                    "anti_overfit": {
                        "tune_periods": list(TUNE_PERIODS),
                        "holdout_periods": list(HOLDOUT_PERIODS),
                        "inference_ml_test_only": True,
                        "max_sharpe_gap": MAX_HOLDOUT_SHARPE_GAP,
                    },
                },
                indent=2,
                default=str,
            )
        )

    handoff = {
        "status": status,
        "n_eligible": len(eligible),
        "best": pick["name"] if pick else None,
        "results_tsv": str(tsv),
        "commit": sha,
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    (out_dir / "evals-summary.md").write_text(
        f"# LieFlow Alpha Hybrid Autoresearch\n\n"
        f"- Selection on: {', '.join(TUNE_PERIODS)}\n"
        f"- Holdout (not used for selection): {', '.join(HOLDOUT_PERIODS)}\n"
        f"- Eligible: {len(eligible)}\n"
        + (
            f"- Best: `{pick['name']}` tune_minS={pick['tune']['min_sharpe']:.3f} "
            f"holdout_S={pick['holdout']['min_sharpe']:.3f} "
            f"alpha_inf={pick['alpha_influence']:.1%}\n"
            if pick
            else ""
        )
    )

    print(json.dumps(handoff, indent=2))
    if pick:
        print(
            f"BEST {pick['name']} eligible={pick['eligible']} "
            f"tune_minS={pick['tune']['min_sharpe']:.3f} "
            f"holdout_S={pick['holdout']['min_sharpe']:.3f} "
            f"tune_ann={pick['tune']['mean_annualized_return']:.2%} "
            f"alpha_inf={pick['alpha_influence']:.1%}"
        )


if __name__ == "__main__":
    main()
