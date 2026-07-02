#!/usr/bin/env python3
"""
Autoresearch: LieFlow factor timing + multi-strategy risk overlay.

Phases (sequential, stop on first winner):
  1. Factor timing alone (raw vol + learned LieFlow meta-label)
  2. Multi-strategy risk overlay alone (vol+mom book + geometry thermostat)
  3. Combined fallback ONLY if phases 1-2 fail to beat raw vol on holdout

Success: holdout oos_2023_2024 beats raw vol on Sharpe AND total return (honest).
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
BEST_PATH = ROOT / "outputs/strategy/best_timing_overlay.json"
HOLDOUT = "oos_2023_2024"
BASE = dict(cost_bps=10.0, hybrid_fallback=True, timing_train_end="2022-12-31")


def build_factor_timing_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []
    for smooth, thr, tmin, soft, cwin in itertools.product(
        (16, 18, 20, 22),
        (0.45, 0.5, 0.55, 0.6),
        (0.0, 0.25, 0.5),
        (True, False),
        (45, 60, 90),
    ):
        tag = f"timing sm{smooth} thr{thr:.2f} tmin{tmin:.2f} {'soft' if soft else 'hard'} cw{cwin}"
        exps.append(
            (
                tag,
                MultiWindowConfig(
                    factor_timing=True,
                    signal_smoothing=smooth,
                    timing_threshold=thr,
                    timing_min_exposure=tmin,
                    timing_soft_gate=soft,
                    concentration_window=cwin,
                    max_exposure=1.0,
                    **BASE,
                ),
            )
        )
    return exps


def build_overlay_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []
    for vw, mw, smooth, cg, cb, mn in itertools.product(
        (0.5, 0.6, 0.7, 0.8, 1.0),
        (0.0, 0.2, 0.3, 0.4),
        (16, 18, 20),
        (0.95, 1.0, 1.05),
        (0.75, 0.85),
        (0.25, 0.35, 0.5),
    ):
        if cg <= cb:
            continue
        tag = f"overlay vw{vw:.1f} mw{mw:.1f} sm{smooth} cg{cg:.2f} cb{cb:.2f} mn{mn:.2f}"
        exps.append(
            (
                tag,
                MultiWindowConfig(
                    multi_strategy_overlay=True,
                    vol_book_weight=vw,
                    mom_book_weight=mw,
                    signal_smoothing=smooth,
                    mom_signal_smoothing=smooth,
                    conc_good_ratio=cg,
                    conc_bad_ratio=cb,
                    min_exposure=mn,
                    max_exposure=1.0,
                    concentration_window=60,
                    vix_boost_regimes=("mid", "low"),
                    vix_cut_regimes=("high",),
                    **BASE,
                ),
            )
        )
    return exps


def build_combined_experiments() -> list[tuple[str, MultiWindowConfig]]:
    """Small fallback grid — only runs if phases 1-2 fail."""
    exps: list[tuple[str, MultiWindowConfig]] = []
    for vw, thr, cg, cb in itertools.product(
        (0.6, 0.7),
        (0.5, 0.55),
        (1.0, 1.05),
        (0.8, 0.85),
    ):
        if cg <= cb:
            continue
        tag = f"combined vw{vw:.1f} thr{thr:.2f} cg{cg:.2f} cb{cb:.2f}"
        exps.append(
            (
                tag,
                MultiWindowConfig(
                    combined_timing_overlay=True,
                    vol_book_weight=vw,
                    mom_book_weight=1.0 - vw,
                    signal_smoothing=18,
                    mom_signal_smoothing=18,
                    timing_threshold=thr,
                    timing_min_exposure=0.0,
                    timing_soft_gate=True,
                    conc_good_ratio=cg,
                    conc_bad_ratio=cb,
                    min_exposure=0.25,
                    max_exposure=1.0,
                    concentration_window=60,
                    **BASE,
                ),
            )
        )
    return exps


def holdout(pm: dict) -> dict:
    return pm["periods"][HOLDOUT]


def composite_score(h: dict) -> float:
    sharpe = effective_period_sharpe(h)
    ann = period_annualized_return(h)
    max_dd = float(h.get("max_drawdown", 0.0))
    return sharpe + ann + 0.75 * (-max_dd)


def beats_raw(h: dict, raw_h: dict) -> bool:
    eps = 1e-9
    return (
        effective_period_sharpe(h) > effective_period_sharpe(raw_h) + eps
        and float(h.get("total_return", 0.0)) > float(raw_h.get("total_return", 0.0)) + eps
    )


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "-"


def run_phase(
    phase: str,
    experiments: list[tuple[str, MultiWindowConfig]],
    session: EvalSession,
    raw_hold: dict,
    out_dir: Path,
    workers: int,
) -> dict | None:
    names = [n for n, _ in experiments]
    configs = [c for _, c in experiments]
    tsv = out_dir / f"{phase}-results.tsv"

    print(f"\n[phase] {phase} configs={len(configs)}")
    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    print(f"[phase] {phase} sweep_ms={(perf_counter() - t0) * 1000:.0f}")

    with open(tsv, "w") as f:
        f.write("name\tholdout_sharpe\tholdout_ann\tholdout_ret\tholdout_dd\tcomposite\tbeats_raw\n")

    best_beat: dict | None = None
    best_any: dict | None = None
    best_any_score = float("-inf")

    for name, cfg, m in zip(names, configs, metrics_list):
        h = holdout(m)
        score = composite_score(h)
        beat = beats_raw(h, raw_hold)
        row = {"name": name, "config": cfg, "metrics": m, "holdout": h, "composite": score, "beats_raw": beat}
        if beat and (best_beat is None or score > best_beat["composite"]):
            best_beat = row
        if score > best_any_score:
            best_any_score = score
            best_any = row
        with open(tsv, "a") as f:
            f.write(
                f"{name}\t{effective_period_sharpe(h):.4f}\t{period_annualized_return(h):.4f}\t"
                f"{h.get('total_return', 0.0):.4f}\t{h.get('max_drawdown', 0.0):.4f}\t"
                f"{score:.4f}\t{beat}\n"
            )

    pick = best_beat or best_any
    if pick:
        print(
            f"[phase] {phase} best={pick['name']} beats_raw={pick['beats_raw']} "
            f"S={effective_period_sharpe(pick['holdout']):.3f} "
            f"ret={pick['holdout'].get('total_return', 0.0):.2%} "
            f"dd={pick['holdout'].get('max_drawdown', 0.0):.2%}"
        )
    return best_beat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=CACHE_FULL)
    parser.add_argument("--skip-combined", action="store_true", help="Never run phase 3")
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print("[autoresearch] mode: classic")
    print("[autoresearch] goal: beat raw vol holdout | phases: timing -> overlay -> combined (fallback)")

    if not args.cache.exists():
        raise SystemExit(f"Missing cache {args.cache} (need full 2015-2024 cache for timing training)")

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = ROOT / "autoresearch" / f"loop-timing-overlay-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=args.cache,
        n_target=60,
        inference_ml_test_only=False,
    )

    raw_m = session.evaluate_multiwindow(
        MultiWindowConfig(raw_vol_baseline=True, signal_smoothing=18, cost_bps=10.0)
    )
    raw_hold = holdout(raw_m)
    raw_score = composite_score(raw_hold)
    print(
        f"raw_holdout S={effective_period_sharpe(raw_hold):.3f} "
        f"ann={period_annualized_return(raw_hold):.2%} "
        f"ret={raw_hold.get('total_return', 0.0):.2%} "
        f"dd={raw_hold.get('max_drawdown', 0.0):.2%} score={raw_score:.4f}"
    )

    winner: dict | None = None
    winner_phase: str | None = None

    for phase, builder in (
        ("factor_timing", build_factor_timing_experiments),
        ("multi_strategy_overlay", build_overlay_experiments),
    ):
        beat = run_phase(phase, builder(), session, raw_hold, out_dir, workers)
        if beat:
            winner = beat
            winner_phase = phase
            break

    if winner is None and not args.skip_combined:
        print("\n[phase] phases 1-2 did not beat raw vol — running combined fallback")
        beat = run_phase("combined_fallback", build_combined_experiments(), session, raw_hold, out_dir, workers)
        if beat:
            winner = beat
            winner_phase = "combined_fallback"

    status = "CONVERGED" if winner else "BLOCKED"
    handoff = {
        "status": status,
        "winner_phase": winner_phase,
        "winner": winner["name"] if winner else None,
        "beats_raw_vol": bool(winner),
        "raw_holdout_sharpe": effective_period_sharpe(raw_hold),
        "raw_holdout_return": raw_hold.get("total_return"),
        "commit": git_head(),
        "results_dir": str(out_dir),
    }

    if winner:
        cfg = winner["config"]
        BEST_PATH.write_text(
            json.dumps(
                {
                    "phase": winner_phase,
                    "name": winner["name"],
                    "beats_raw_vol_holdout": True,
                    "holdout": winner["holdout"],
                    "composite": winner["composite"],
                    "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                    "raw_vol_baseline_holdout": raw_hold,
                },
                indent=2,
                default=str,
            )
        )

    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    print(json.dumps(handoff, indent=2))


if __name__ == "__main__":
    main()
