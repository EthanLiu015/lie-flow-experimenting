#!/usr/bin/env python3
"""
Autoresearch: honest LieFlow strategies with pre-2023-trained model.

Pipeline:
- LieFlow trained on clouds_pre2023.npy only (never saw 2023+)
- Inference cache contains OOS 2023+ dates only
- Pre-2023: raw-vol fallback; 2023+: LieFlow active

Anti-overfitting:
- Select on HOLDOUT (oos_2023_2024) only
- Eligibility requires beating raw-vol baseline on holdout (Sharpe AND return)
- Small focused grid (~60 configs)
- Report full 6-window honest metrics separately
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
    lieflow_influence_fraction,
)
from lieflow_quant.session import EvalSession, MultiWindowConfig
from lieflow_quant.validation import build_raw_vol_signals_for_sections

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs/strategy/best_lieflow_honest.json"
CACHE_PATH = ROOT / "outputs/strategy/inference_cache_pre2023train_oos2023.npz"

HOLDOUT = ("oos_2023_2024",)
FULL_PERIODS = (
    "is_2015_2022",
    "oos_2023_2024",
    "wf_2015_2017",
    "wf_2018_2019",
    "wf_2020_2021",
    "wf_2022_2024",
)


def build_experiments() -> list[tuple[str, MultiWindowConfig]]:
    """Focused grid — avoid large sweeps that overfit holdout."""
    exps: list[tuple[str, MultiWindowConfig]] = []
    base = dict(cost_bps=10.0, hybrid_fallback=True, min_exposure=0.5, max_exposure=1.0)

    # 1. LieFlow alpha blend
    for strat in ("mom_minus_vol", "mom_resid_vol", "canonical_vol", "conc_scaled_momentum"):
        for sign in (-1.0, 1.0) if strat in ("mom_minus_vol", "canonical_vol") else (1.0,):
            for lw in (0.3, 0.45, 0.6):
                for smooth in (16, 20):
                    exps.append(
                        (
                            f"alpha {strat} s{int(sign)} w{lw:.2f} sm{smooth}",
                            MultiWindowConfig(
                                hybrid_alpha=True,
                                strategy=strat,
                                signal_sign=sign,
                                lieflow_weight=lw,
                                signal_smoothing=smooth,
                                **base,
                            ),
                        )
                    )

    # 2. Adaptive concentration-weighted alpha
    for strat in ("mom_minus_vol", "mom_resid_vol"):
        for lw in (0.4, 0.55):
            exps.append(
                (
                    f"adaptive {strat} w{lw:.2f} sm18",
                    MultiWindowConfig(
                        hybrid_adaptive=True,
                        strategy=strat,
                        lieflow_weight=lw,
                        signal_smoothing=18,
                        **base,
                    ),
                )
            )

    # 3. LieFlow concentration gate + raw vol (filter role)
    for conc in (0.5, 0.7, 0.9):
        for lw in (0.25, 0.4):
            exps.append(
                (
                    f"gate c{conc:.1f} w{lw:.2f} sm18",
                    MultiWindowConfig(
                        hybrid_vol=True,
                        lieflow_weight=lw,
                        signal_smoothing=18,
                        min_concentration_ratio=conc,
                        **base,
                    ),
                )
            )

    # 4. Full LieFlow alpha on OOS (w=1.0 blend = pure LieFlow when available)
    for strat in ("mom_resid_vol", "radial_distance", "mom_minus_vol"):
        exps.append(
            (
                f"pure_alpha {strat} sm20",
                MultiWindowConfig(
                    hybrid_alpha=True,
                    strategy=strat,
                    lieflow_weight=1.0,
                    signal_smoothing=20,
                    **base,
                ),
            ),
        )

    return exps


def holdout_metrics(full: dict) -> dict[str, float]:
    return aggregate_multiwindow_metrics(full["periods"], HOLDOUT)


def full_metrics(full: dict) -> dict[str, float]:
    return aggregate_multiwindow_metrics(full["periods"], FULL_PERIODS)


def beats_baseline(hold: dict[str, float], raw_hold: dict[str, float]) -> bool:
    return (
        hold["min_sharpe"] >= raw_hold["min_sharpe"]
        and hold["mean_total_return"] >= raw_hold["mean_total_return"]
    )


def selection_score(hold: dict[str, float]) -> float:
    return hold["min_sharpe"] + hold["mean_annualized_return"]


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print("[autoresearch] mode: classic")
    print("[autoresearch] metric: holdout sharpe+return | pre-2023-trained LieFlow | beat raw vol")

    if not args.cache.exists():
        raise SystemExit(f"Missing cache {args.cache}; run cache_inference on pre-2023-trained checkpoint first.")

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = ROOT / "autoresearch" / f"loop-lieflow-honest-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    experiments = build_experiments()
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
    raw_hold = holdout_metrics(raw_m)
    raw_full = full_metrics(raw_m)
    print(
        f"configs={len(configs)} cache_rows={len(session.cache)} "
        f"raw_holdout S={raw_hold['min_sharpe']:.3f} ret={raw_hold['mean_total_return']:.2%} "
        f"raw_full minS={raw_full['min_sharpe']:.3f}"
    )

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    print(f"sweep_ms={(perf_counter() - t0) * 1000:.0f}")

    with open(tsv, "w") as f:
        f.write(
            "name\tholdout_sharpe\tholdout_ret\tfull_min_sharpe\tfull_mean_ann\t"
            "alpha_inf\tbeats_raw\tselection\teligible\n"
        )

    eligible: list[dict] = []
    best: dict | None = None
    best_score = float("-inf")

    for name, cfg, m in zip(names, configs, metrics_list):
        hold = holdout_metrics(m)
        full = full_metrics(m)
        sig = session.build_signals(cfg)
        cov = coverage_fraction(sig, session.section_dates)
        if cfg.hybrid_alpha or cfg.hybrid_adaptive:
            influence = lieflow_alpha_influence_fraction(sig, raw_vol, lieflow_dates)
        else:
            influence = lieflow_influence_fraction(sig, lieflow_dates)
        beat = beats_baseline(hold, raw_hold)
        score = selection_score(hold)
        ok = beat and influence >= 0.05
        row = {
            "name": name,
            "config": cfg,
            "metrics": m,
            "holdout": hold,
            "full": full,
            "influence": influence,
            "beats_raw": beat,
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
                f"{name}\t{hold['min_sharpe']:.4f}\t{hold['mean_total_return']:.4f}\t"
                f"{full['min_sharpe']:.4f}\t{full['mean_annualized_return']:.4f}\t"
                f"{influence:.4f}\t{beat}\t{score:.4f}\t{ok}\n"
            )

    eligible.sort(key=lambda r: (-r["selection"], -r["full"]["min_sharpe"]))
    pick = eligible[0] if eligible else best
    status = "CONVERGED" if eligible else "BOUNDED"

    if pick:
        cfg = pick["config"]
        BEST_PATH.write_text(
            json.dumps(
                {
                    "name": pick["name"],
                    "eligible": pick["eligible"],
                    "beats_raw_vol_holdout": pick["beats_raw"],
                    "selection_score": pick["selection"],
                    "influence": pick["influence"],
                    "config": {
                        "hybrid_alpha": cfg.hybrid_alpha,
                        "hybrid_adaptive": cfg.hybrid_adaptive,
                        "hybrid_vol": cfg.hybrid_vol,
                        "strategy": cfg.strategy,
                        "lieflow_weight": cfg.lieflow_weight,
                        "signal_smoothing": cfg.signal_smoothing,
                        "min_concentration_ratio": cfg.min_concentration_ratio,
                    },
                    "holdout_metrics": pick["holdout"],
                    "full_metrics": pick["full"],
                    "raw_vol_baseline": {"holdout": raw_hold, "full": raw_full},
                    "methodology": {
                        "lieflow_training": "clouds_pre2023.npy only (temporal 95/5 val)",
                        "inference_cache": str(args.cache),
                        "selection": "holdout oos_2023_2024 only",
                        "eligibility": "beat raw vol on holdout sharpe AND return",
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
        "commit": git_head(),
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    print(json.dumps(handoff, indent=2))
    if pick:
        print(
            f"BEST {pick['name']} eligible={pick['eligible']} "
            f"holdout_S={pick['holdout']['min_sharpe']:.3f} "
            f"holdout_ret={pick['holdout']['mean_total_return']:.2%} "
            f"full_minS={pick['full']['min_sharpe']:.3f} "
            f"beats_raw={pick['beats_raw']}"
        )


if __name__ == "__main__":
    main()
