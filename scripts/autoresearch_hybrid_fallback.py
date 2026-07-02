#!/usr/bin/env python3
"""Autoresearch: hybrid with LieFlow fallback — honest Sharpe + annual return."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

from lieflow_quant.methodology import coverage_fraction, lieflow_influence_fraction
from lieflow_quant.session import EvalSession, MultiWindowConfig

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs/strategy/best_hybrid_fallback.json"

SHARPE_FLOOR = 0.30
MIN_LIEFLOW_WEIGHT = 0.2
MIN_LIEFLOW_INFLUENCE = 0.05


def build_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []
    for conc in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.02):
        for smooth in (14, 16, 18, 20, 22):
            for lw in (0.2, 0.3, 0.4, 0.5):
                for min_exp, max_exp in ((0.5, 1.0), (0.75, 1.0)):
                    exps.append(
                        (
                            f"hybrid fb c{conc:.2f} s{smooth} w{lw:.1f} e{min_exp:.2f}-{max_exp:.2f}",
                            MultiWindowConfig(
                                hybrid_vol=True,
                                hybrid_fallback=True,
                                min_concentration_ratio=conc if conc > 0 else None,
                                signal_smoothing=smooth,
                                lieflow_weight=lw,
                                cost_bps=10.0,
                                min_exposure=min_exp,
                                max_exposure=max_exp,
                            ),
                        )
                    )
    return exps


def composite_score(
    metrics: dict,
    *,
    lieflow_influence: float,
    sharpe_floor: float,
) -> float:
    min_s = float(metrics["min_sharpe"])
    mean_ann = float(metrics["mean_annualized_return"])
    mean_s = float(metrics["mean_sharpe"])
    if min_s < sharpe_floor:
        return -100.0 + min_s
    if lieflow_influence < MIN_LIEFLOW_INFLUENCE:
        return -50.0 + lieflow_influence
    return mean_ann + 0.5 * min_s + 0.25 * mean_s


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
        f"[autoresearch] metric: mean_ann_return + sharpe | "
        f"hybrid_fallback=True | ml_test_only=True | cost=10bps"
    )

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = args.output_dir or ROOT / "autoresearch" / f"loop-hybrid-fallback-{ts}"
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
    lieflow_dates = frozenset(session.cache["date"].dt.normalize())

    raw_baseline = session.evaluate_multiwindow(
        MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=18, cost_bps=10.0)
    )
    print(
        f"configs={len(configs)} workers={workers} ml_start={session.ml_test_start.date()} "
        f"raw_baseline minS={raw_baseline['min_sharpe']:.3f} "
        f"meanAnn={raw_baseline['mean_annualized_return']:.2%}"
    )

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    print(f"sweep_ms={(perf_counter() - t0) * 1000:.0f}")

    with open(tsv, "w") as f:
        f.write(
            "name\tmin_sharpe\tmean_sharpe\tmean_ann_ret\tmean_total_ret\t"
            "lieflow_influence\tcoverage\tcomposite\teligible\n"
        )

    best_score = float("-inf")
    best: dict | None = None
    eligible: list[dict] = []
    sha = git_head()

    for name, cfg, m in zip(names, configs, metrics_list):
        if cfg.lieflow_weight < MIN_LIEFLOW_WEIGHT:
            continue
        sig = session.build_signals(cfg)
        cov = coverage_fraction(sig, session.section_dates)
        influence = lieflow_influence_fraction(sig, lieflow_dates)
        score = composite_score(m, lieflow_influence=influence, sharpe_floor=args.sharpe_floor)
        ok = (
            float(m["min_sharpe"]) >= args.sharpe_floor
            and influence >= MIN_LIEFLOW_INFLUENCE
            and cfg.hybrid_vol
        )
        row = {
            "name": name,
            "config": cfg,
            "metrics": m,
            "coverage": cov,
            "lieflow_influence": influence,
            "composite": score,
            "eligible": ok,
        }
        if ok:
            eligible.append(row)
        if score > best_score:
            best_score = score
            best = row

        with open(tsv, "a") as f:
            f.write(
                f"{name}\t{m['min_sharpe']:.4f}\t{m['mean_sharpe']:.4f}\t"
                f"{m['mean_annualized_return']:.4f}\t{m['mean_total_return']:.4f}\t"
                f"{influence:.4f}\t{cov:.2%}\t{score:.4f}\t{ok}\n"
            )

    eligible.sort(key=lambda r: (-r["composite"], -r["metrics"]["mean_annualized_return"]))
    pick = eligible[0] if eligible else best
    status = "CONVERGED" if eligible else "BOUNDED"

    if pick:
        cfg = pick["config"]
        BEST_PATH.write_text(
            json.dumps(
                {
                    "name": pick["name"],
                    "eligible": pick["eligible"],
                    "composite": pick["composite"],
                    "lieflow_influence": pick["lieflow_influence"],
                    "coverage": pick["coverage"],
                    "config": {
                        "hybrid_vol": cfg.hybrid_vol,
                        "hybrid_fallback": cfg.hybrid_fallback,
                        "min_concentration_ratio": cfg.min_concentration_ratio,
                        "lieflow_weight": cfg.lieflow_weight,
                        "signal_smoothing": cfg.signal_smoothing,
                        "cost_bps": cfg.cost_bps,
                        "min_exposure": cfg.min_exposure,
                        "max_exposure": cfg.max_exposure,
                    },
                    "metrics": pick["metrics"],
                    "raw_vol_baseline": {
                        "min_sharpe": raw_baseline["min_sharpe"],
                        "mean_annualized_return": raw_baseline["mean_annualized_return"],
                    },
                    "methodology": [
                        "calendar-filled Sharpe (flat days = 0% return, Sharpe=0 in aggregates)",
                        "LieFlow inference ML test split only",
                        "pre-OOS dates fall back to raw vol at full exposure",
                        "10bps costs",
                    ],
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
    summary = out_dir / "evals-summary.md"
    summary.write_text(
        f"# Hybrid Fallback Autoresearch\n\n"
        f"- Honest min/mean Sharpe across all 6 windows\n"
        f"- ML test-only LieFlow; IS dates use raw-vol fallback\n"
        f"- Cost: 10bps\n"
        f"- Eligible (min_sharpe≥{args.sharpe_floor}, lieflow_influence≥{MIN_LIEFLOW_INFLUENCE}): {len(eligible)}\n"
        + (
            f"- Best: `{pick['name']}` composite={pick['composite']:.4f} "
            f"minS={pick['metrics']['min_sharpe']:.3f} "
            f"meanAnn={pick['metrics']['mean_annualized_return']:.2%} "
            f"influence={pick['lieflow_influence']:.1%}\n"
            if pick
            else ""
        )
        + (
            f"- Raw vol baseline: minS={raw_baseline['min_sharpe']:.3f} "
            f"meanAnn={raw_baseline['mean_annualized_return']:.2%}\n"
        )
    )

    print(json.dumps(handoff, indent=2))
    if pick:
        m = pick["metrics"]
        print(
            f"BEST {pick['name']} eligible={pick['eligible']} "
            f"minS={m['min_sharpe']:.3f} meanAnn={m['mean_annualized_return']:.2%} "
            f"influence={pick['lieflow_influence']:.1%} cov={pick['coverage']:.1%}"
        )
    if eligible[:3]:
        print("Top eligible:")
        for r in eligible[:3]:
            print(
                f"  {r['name']} minS={r['metrics']['min_sharpe']:.3f} "
                f"meanAnn={r['metrics']['mean_annualized_return']:.2%}"
            )


if __name__ == "__main__":
    main()
