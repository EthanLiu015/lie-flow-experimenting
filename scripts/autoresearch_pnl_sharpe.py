#!/usr/bin/env python3
"""Autoresearch: maximize PnL (total return) while enforcing a Sharpe floor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from lieflow_quant.session import EvalSession, MultiWindowConfig, config_from_argv
from lieflow_quant.methodology import coverage_fraction

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs/strategy/best_pnl_sharpe_config.json"

# Bias toward full-coverage / higher-participation configs (PnL-friendly).
EXPERIMENTS: list[tuple[str, list[str]]] = [
    ("raw_vol s10", ["--raw-vol-baseline", "--signal-smoothing", "10"]),
    ("raw_vol s15", ["--raw-vol-baseline", "--signal-smoothing", "15"]),
    ("raw_vol s20", ["--raw-vol-baseline", "--signal-smoothing", "20"]),
    ("raw_vol s25", ["--raw-vol-baseline", "--signal-smoothing", "25"]),
    ("raw_vol s20 cost5", ["--raw-vol-baseline", "--signal-smoothing", "20", "--cost-bps", "5"]),
    ("hybrid conc0.9 s20", ["--hybrid-vol", "--min-concentration-ratio", "0.9", "--signal-smoothing", "20"]),
    ("hybrid conc1.0 s10", ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "10"]),
    ("hybrid conc1.0 s15", ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "15"]),
    ("hybrid conc1.0 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "20"]),
    ("hybrid conc1.0 s25", ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "25"]),
    ("hybrid conc1.0 cost5 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.0", "--signal-smoothing", "20", "--cost-bps", "5"]),
    ("hybrid conc1.05 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.05", "--signal-smoothing", "20"]),
    ("hybrid conc1.1 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.1", "--signal-smoothing", "20"]),
    ("hybrid conc1.2 w0.3 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.2", "--lieflow-weight", "0.3", "--signal-smoothing", "20"]),
    ("hybrid conc1.2 w0.5 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.2", "--lieflow-weight", "0.5", "--signal-smoothing", "20"]),
    ("hybrid conc1.2 w0.7 s20", ["--hybrid-vol", "--min-concentration-ratio", "1.2", "--lieflow-weight", "0.7", "--signal-smoothing", "20"]),
    ("canonical_vol s15", ["--strategy", "canonical_vol", "--signal-smoothing", "15"]),
    ("canonical_vol s20", ["--strategy", "canonical_vol", "--signal-smoothing", "20"]),
    ("canonical_vol s25", ["--strategy", "canonical_vol", "--signal-smoothing", "25"]),
    ("mom_minus_vol flip s25", ["--strategy", "mom_minus_vol", "--signal-sign", "-1", "--signal-smoothing", "25"]),
]

SHARPE_FLOOR = 0.30


def mean_total_return(metrics: dict) -> float:
    periods = metrics.get("periods", {})
    rets = [float(pm["total_return"]) for pm in periods.values() if pm.get("n_days", 0) > 0]
    return sum(rets) / len(rets) if rets else float("nan")


def min_total_return(metrics: dict) -> float:
    periods = metrics.get("periods", {})
    rets = [float(pm["total_return"]) for pm in periods.values() if pm.get("n_days", 0) > 0]
    return min(rets) if rets else float("nan")


def composite_score(metrics: dict, *, sharpe_floor: float = SHARPE_FLOOR) -> float:
    """
    Primary: mean total return across 6 windows.
    Guard: min Sharpe must clear floor (else heavy penalty).
    Tiebreaker: min Sharpe and min window return.
    """
    min_s = float(metrics["min_sharpe"])
    mean_ret = mean_total_return(metrics)
    min_ret = min_total_return(metrics)
    if min_s < sharpe_floor:
        return -100.0 + min_s
    return mean_ret + 0.05 * min_s + 0.02 * min_ret


def coverage_pct(session: EvalSession, config: MultiWindowConfig, signals: pd.DataFrame) -> float:
    return coverage_fraction(signals, session.section_dates)


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
    print(f"[autoresearch] metric: composite_pnl_sharpe (mean_return + sharpe_floor={args.sharpe_floor})")

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = args.output_dir or ROOT / "autoresearch" / f"loop-pnl-sharpe-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        n_target=60,
    )
    print(f"Session ready in {session.load_ms:.0f}ms")

    names = [n for n, _ in EXPERIMENTS]
    configs = [config_from_argv(argv) for _, argv in EXPERIMENTS]

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    sweep_ms = (perf_counter() - t0) * 1000
    print(f"Sweep finished in {sweep_ms:.0f}ms ({sweep_ms / len(configs):.0f}ms/config)")

    with open(tsv, "w") as f:
        f.write("# metric_direction: higher_is_better (composite_pnl_sharpe)\n")
        f.write(
            "name\tmin_sharpe\tmean_sharpe\tmean_return\tmin_return\tcoverage\tcomposite\tstatus\n"
        )

    best_score = float("-inf")
    best_name = ""
    best_metrics: dict | None = None
    best_args: list[str] = []
    best_coverage = 0.0
    sha = git_head()

    rows: list[dict] = []
    for name, argv, m, cfg in zip(names, [a for _, a in EXPERIMENTS], metrics_list, configs):
        score = composite_score(m, sharpe_floor=args.sharpe_floor)
        m_ret = mean_total_return(m)
        mn_ret = min_total_return(m)
        sig = session.build_signals(cfg)
        cov = coverage_pct(session, cfg, sig)
        status = "keep" if score >= best_score else "discard"
        if score >= best_score:
            best_score = score
            best_name = name
            best_metrics = m
            best_args = argv
            best_coverage = cov

        rows.append(
            {
                "name": name,
                "args": argv,
                "min_sharpe": float(m["min_sharpe"]),
                "mean_sharpe": float(m["mean_sharpe"]),
                "mean_total_return": m_ret,
                "min_total_return": mn_ret,
                "coverage": cov,
                "composite": score,
                "all_windows_positive": m["all_windows_positive"],
            }
        )

        with open(tsv, "a") as f:
            f.write(
                f"{name}\t{m['min_sharpe']:.4f}\t{m['mean_sharpe']:.4f}\t"
                f"{m_ret:.4f}\t{mn_ret:.4f}\t{cov:.2%}\t{score:.4f}\t{status}\n"
            )

        print(
            f"{name:28s} minS={m['min_sharpe']:6.3f} meanRet={m_ret:6.3f} "
            f"minRet={mn_ret:6.3f} cov={cov:5.1%} score={score:6.3f} ({status})"
        )

    if best_metrics:
        BEST_PATH.write_text(
            json.dumps(
                {
                    "name": best_name,
                    "args": best_args,
                    "composite_score": best_score,
                    "coverage": best_coverage,
                    "sharpe_floor": args.sharpe_floor,
                    "metrics": best_metrics,
                },
                indent=2,
            )
        )

    # Also report best by raw mean return among configs meeting Sharpe floor
    eligible = [r for r in rows if r["min_sharpe"] >= args.sharpe_floor]
    best_pnl = max(eligible, key=lambda r: r["mean_total_return"]) if eligible else None

    summary = out_dir / "evals-summary.md"
    summary.write_text(
        f"# PnL + Sharpe Autoresearch\n\n"
        f"- **Metric:** composite = mean_total_return + 0.05×min_sharpe + 0.02×min_window_return\n"
        f"- **Sharpe floor:** {args.sharpe_floor}\n"
        f"- **Best composite:** `{best_name}` (score={best_score:.4f}, coverage={best_coverage:.1%})\n"
        f"- **Best min Sharpe:** {best_metrics['min_sharpe']:.4f}\n"
        f"- **Best mean return:** {mean_total_return(best_metrics):.4f}\n"
        + (
            f"- **Highest PnL (eligible):** `{best_pnl['name']}` "
            f"(mean_ret={best_pnl['mean_total_return']:.4f}, min_sharpe={best_pnl['min_sharpe']:.4f}, "
            f"coverage={best_pnl['coverage']:.1%})\n"
            if best_pnl
            else ""
        )
        + f"\n## Recommendation\n\n"
        f"For **max PnL with Sharpe discipline**, prefer full-coverage configs with "
        f"min_sharpe ≥ {args.sharpe_floor} rather than selective high-Sharpe/low-coverage gates.\n"
    )

    handoff = {
        "version": "2.1.0",
        "source": "loop-pnl-sharpe",
        "timestamp": datetime.now().isoformat(),
        "status": "CONVERGED" if best_metrics and best_metrics["min_sharpe"] >= args.sharpe_floor else "BOUNDED",
        "results_tsv": str(tsv),
        "commit": sha,
        "findings": [
            f"best_composite={best_name}",
            f"composite_score={best_score:.4f}",
            f"min_sharpe={best_metrics['min_sharpe']:.4f}" if best_metrics else "no_winner",
            f"mean_total_return={mean_total_return(best_metrics):.4f}" if best_metrics else "no_winner",
            f"coverage={best_coverage:.2%}",
            f"best_pnl_eligible={best_pnl['name'] if best_pnl else 'none'}",
        ],
        "config": {
            "goal": "maximize PnL while caring about Sharpe",
            "metric": "composite_pnl_sharpe",
            "sharpe_floor": args.sharpe_floor,
            "direction": "higher_is_better",
        },
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))

    print(f"\nBEST (composite): {best_name} score={best_score:.4f} coverage={best_coverage:.1%}")
    if best_pnl and best_pnl["name"] != best_name:
        print(
            f"BEST (PnL eligible): {best_pnl['name']} "
            f"mean_ret={best_pnl['mean_total_return']:.4f} min_sharpe={best_pnl['min_sharpe']:.4f}"
        )
    print(f"Saved -> {BEST_PATH}")
    print(json.dumps(handoff, indent=2))


if __name__ == "__main__":
    main()
