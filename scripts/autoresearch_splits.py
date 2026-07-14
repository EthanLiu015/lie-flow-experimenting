#!/usr/bin/env python3
"""
Autoresearch: sweep long/short portfolio splits.

Evaluates different top_n / bottom_n configurations for the raw vol
strategy. The default 30/30 (=0/0) uses all 60 stocks; narrower
splits concentrate in the strongest-signal names.

Success: holdout oos_2023_2024 beats current best (30/30 with sm22).
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

from lieflow_quant.methodology import (
    effective_period_sharpe,
    period_annualized_return,
)
from lieflow_quant.session import EvalSession, MultiWindowConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE_FULL = ROOT / "outputs/strategy/inference_cache_full_n60.npz"
HOLDOUT = "oos_2023_2024"
BASE = dict(cost_bps=10.0, hybrid_fallback=True)


def build_split_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []

    symmetric = [0, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28]
    asymmetric = [
        (10, 20), (20, 10),
        (15, 25), (25, 15),
        (10, 15), (15, 10),
        (20, 25), (25, 20),
        (5, 10), (10, 5),
        (5, 15), (15, 5),
    ]

    for smooth in (20, 22, 24):
        for n in symmetric:
            tag = f"raw sm{smooth} L{n}S{n}" if n > 0 else f"raw sm{smooth} L30S30"
            exps.append((
                tag,
                MultiWindowConfig(
                    raw_vol_baseline=True,
                    signal_smoothing=smooth,
                    long_n=n,
                    short_n=n,
                    **BASE,
                ),
            ))
        for ln, sn in asymmetric:
            tag = f"raw sm{smooth} L{ln}S{sn}"
            exps.append((
                tag,
                MultiWindowConfig(
                    raw_vol_baseline=True,
                    signal_smoothing=smooth,
                    long_n=ln,
                    short_n=sn,
                    **BASE,
                ),
            ))

    return exps


def holdout(pm: dict) -> dict:
    return pm["periods"][HOLDOUT]


def composite_score(h: dict) -> float:
    sharpe = effective_period_sharpe(h)
    ann = period_annualized_return(h)
    max_dd = float(h.get("max_drawdown", 0.0))
    return sharpe + ann + 0.75 * (-max_dd)


def git_head() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=CACHE_FULL)
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print("[autoresearch] mode: classic")
    print("[autoresearch] goal: find best long/short split for raw vol strategy")

    if not args.cache.exists():
        raise SystemExit(f"Missing cache {args.cache}")

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = ROOT / "autoresearch" / f"loop-splits-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=args.cache,
        n_target=60,
        inference_ml_test_only=False,
    )

    baseline_m = session.evaluate_multiwindow(
        MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=22, **BASE)
    )
    base_hold = holdout(baseline_m)
    base_score = composite_score(base_hold)
    print(
        f"baseline (30/30 sm22) S={effective_period_sharpe(base_hold):.4f} "
        f"ret={base_hold.get('total_return', 0.0):.2%} "
        f"dd={base_hold.get('max_drawdown', 0.0):.2%} score={base_score:.4f}"
    )

    experiments = build_split_experiments()
    names = [n for n, _ in experiments]
    configs = [c for _, c in experiments]
    tsv = out_dir / "splits-results.tsv"

    print(f"\n[sweep] {len(configs)} configurations")
    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    print(f"[sweep] elapsed_ms={(perf_counter() - t0) * 1000:.0f}")

    with open(tsv, "w") as f:
        f.write("name\tholdout_sharpe\tholdout_ann\tholdout_ret\tholdout_dd\tcomposite\n")

    best: dict | None = None
    best_score = float("-inf")

    for name, cfg, m in zip(names, configs, metrics_list):
        h = holdout(m)
        score = composite_score(h)
        sharpe = effective_period_sharpe(h)

        if score > best_score:
            best_score = score
            best = {
                "name": name,
                "config": cfg,
                "holdout": h,
                "composite": score,
            }

        with open(tsv, "a") as f:
            f.write(
                f"{name}\t{sharpe:.4f}\t{period_annualized_return(h):.4f}\t"
                f"{h.get('total_return', 0.0):.4f}\t{h.get('max_drawdown', 0.0):.4f}\t"
                f"{score:.4f}\n"
            )

    beats = False
    if best:
        bh = best["holdout"]
        beats = (
            effective_period_sharpe(bh) > effective_period_sharpe(base_hold) + 1e-9
            and float(bh.get("total_return", 0.0)) > float(base_hold.get("total_return", 0.0)) + 1e-9
        )
        print(
            f"\n[result] best={best['name']} "
            f"S={effective_period_sharpe(bh):.4f} "
            f"ret={bh.get('total_return', 0.0):.2%} "
            f"dd={bh.get('max_drawdown', 0.0):.2%} "
            f"beats_baseline={'YES' if beats else 'NO'}"
        )

    handoff = {
        "status": "CONVERGED" if beats else "BLOCKED",
        "baseline": "raw sm22 L30S30",
        "baseline_sharpe": effective_period_sharpe(base_hold),
        "baseline_return": base_hold.get("total_return"),
        "winner": best["name"] if best else None,
        "winner_sharpe": effective_period_sharpe(best["holdout"]) if best else None,
        "winner_return": best["holdout"].get("total_return") if best else None,
        "winner_drawdown": best["holdout"].get("max_drawdown") if best else None,
        "beats_baseline": beats,
        "commit": git_head(),
        "results_dir": str(out_dir),
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    print(json.dumps(handoff, indent=2))


if __name__ == "__main__":
    main()
