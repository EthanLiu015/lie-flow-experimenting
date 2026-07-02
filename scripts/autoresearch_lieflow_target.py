#!/usr/bin/env python3
"""Autoresearch: LieFlow-in-loop configs targeting min Sharpe > 0.5 and mean PnL > 18%."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

from lieflow_quant.session import EvalSession, MultiWindowConfig

ROOT = Path(__file__).resolve().parents[1]
BEST_PATH = ROOT / "outputs/strategy/best_lieflow_target_config.json"

SHARPE_MIN = 0.5
PNL_MIN = 0.18


def mean_total_return(metrics: dict) -> float:
    periods = metrics.get("periods", {})
    rets = [float(pm["total_return"]) for pm in periods.values() if pm.get("n_days", 0) > 0]
    return sum(rets) / len(rets) if rets else float("nan")


def meets_target(metrics: dict, *, sharpe_min: float, pnl_min: float) -> bool:
    return float(metrics["min_sharpe"]) > sharpe_min and mean_total_return(metrics) > pnl_min


def distance_to_target(metrics: dict, *, sharpe_min: float, pnl_min: float) -> float:
    """Lower is better. Zero when both constraints satisfied."""
    min_s = float(metrics["min_sharpe"])
    mean_ret = mean_total_return(metrics)
    sharpe_gap = max(0.0, sharpe_min - min_s)
    pnl_gap = max(0.0, pnl_min - mean_ret)
    return sharpe_gap + pnl_gap


def build_experiments() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []

    # Fine hybrid grid around Pareto frontier (conc 1.0–1.05, cost5).
    for conc in (0.95, 0.98, 1.0, 1.01, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05, 1.06, 1.08):
        for smooth in (10, 12, 14, 15, 16, 18, 20, 22, 25):
            for cost in (5.0, 10.0):
                for lw in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                    name = f"hybrid c{conc:.3f} s{smooth} cost{int(cost)} w{lw:.1f}"
                    exps.append(
                        (
                            name,
                            MultiWindowConfig(
                                hybrid_vol=True,
                                min_concentration_ratio=conc,
                                signal_smoothing=smooth,
                                cost_bps=cost,
                                lieflow_weight=lw,
                            ),
                        )
                    )

    # Exposure scaling on hybrid neighborhood.
    for conc in (1.0, 1.02, 1.03, 1.04, 1.05):
        for min_exp, max_exp in ((0.5, 1.0), (0.75, 1.0), (0.5, 1.25), (0.25, 1.25)):
            for smooth in (14, 16, 18, 20):
                for lw in (0.4, 0.5, 0.6):
                    name = f"hybrid c{conc:.2f} s{smooth} exp{min_exp}-{max_exp} w{lw}"
                    exps.append(
                        (
                            name,
                            MultiWindowConfig(
                                hybrid_vol=True,
                                min_concentration_ratio=conc,
                                signal_smoothing=smooth,
                                cost_bps=5.0,
                                lieflow_weight=lw,
                                min_exposure=min_exp,
                                max_exposure=max_exp,
                            ),
                        )
                    )

    # Pure LieFlow strategies with cost5 (full coverage).
    for strategy, sign in (
        ("canonical_vol", 1.0),
        ("canonical_vol", -1.0),
        ("mom_minus_vol", -1.0),
        ("mom_resid_vol", 1.0),
        ("mom_resid_vol", -1.0),
        ("radial_distance", 1.0),
        ("radial_distance", -1.0),
    ):
        for smooth in (15, 20, 25, 30):
            for gate in (None, 1.0, 1.05):
                gate_s = f" g{gate}" if gate else ""
                name = f"{strategy} s{smooth} sign{int(sign)}{gate_s} cost5"
                exps.append(
                    (
                        name,
                        MultiWindowConfig(
                            strategy=strategy,
                            signal_sign=sign,
                            signal_smoothing=smooth,
                            cost_bps=5.0,
                            min_concentration_ratio=gate,
                        ),
                    )
                )

    for strategy in ("radial_gated", "mom_resid_gated", "conc_scaled_momentum"):
        for smooth in (10, 15, 20):
            for gate in (1.0, 1.02, 1.05, 1.08):
                name = f"{strategy} s{smooth} g{gate} cost5"
                exps.append(
                    (
                        name,
                        MultiWindowConfig(
                            strategy=strategy,
                            signal_smoothing=smooth,
                            cost_bps=5.0,
                            min_concentration_ratio=gate,
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
    parser.add_argument("--pnl-min", type=float, default=PNL_MIN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-experiments", type=int, default=None)
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)

    print("[autoresearch] mode: classic")
    print(
        f"[autoresearch] metric: lieflow_in_loop | Verify: min_sharpe>{args.sharpe_min} "
        f"AND mean_return>{args.pnl_min:.0%}"
    )

    ts = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = args.output_dir or ROOT / "autoresearch" / f"loop-lieflow-target-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    experiments = build_experiments()
    if args.max_experiments:
        experiments = experiments[: args.max_experiments]
    names = [n for n, _ in experiments]
    configs = [c for _, c in experiments]

    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        n_target=60,
    )
    print(f"Session ready in {session.load_ms:.0f}ms | {len(configs)} configs")

    t0 = perf_counter()
    metrics_list = session.sweep_multiwindow(configs, n_workers=workers)
    sweep_ms = (perf_counter() - t0) * 1000
    print(f"Sweep finished in {sweep_ms:.0f}ms ({sweep_ms / len(configs):.0f}ms/config)")

    with open(tsv, "w") as f:
        f.write("# metric_direction: higher_is_better (meet sharpe+pnl target)\n")
        f.write("name\tmin_sharpe\tmean_sharpe\tmean_return\tgap_to_target\tmeets_target\n")

    winners: list[dict] = []
    best_gap = float("inf")
    best_row: dict | None = None
    sha = git_head()

    for name, cfg, m in zip(names, configs, metrics_list):
        mean_ret = mean_total_return(m)
        gap = distance_to_target(m, sharpe_min=args.sharpe_min, pnl_min=args.pnl_min)
        ok = meets_target(m, sharpe_min=args.sharpe_min, pnl_min=args.pnl_min)
        row = {
            "name": name,
            "config": cfg,
            "min_sharpe": float(m["min_sharpe"]),
            "mean_sharpe": float(m["mean_sharpe"]),
            "mean_total_return": mean_ret,
            "gap": gap,
            "meets_target": ok,
            "metrics": m,
        }
        if ok:
            winners.append(row)
        if gap < best_gap:
            best_gap = gap
            best_row = row

        with open(tsv, "a") as f:
            f.write(
                f"{name}\t{m['min_sharpe']:.4f}\t{m['mean_sharpe']:.4f}\t"
                f"{mean_ret:.4f}\t{gap:.4f}\t{ok}\n"
            )

    winners.sort(key=lambda r: (-r["mean_total_return"], -r["min_sharpe"]))
    if winners:
        pick = winners[0]
        status = "CONVERGED"
    elif best_row:
        pick = best_row
        status = "BOUNDED"
    else:
        pick = None
        status = "FAILED"

    if pick:
        cfg = pick["config"]
        BEST_PATH.write_text(
            json.dumps(
                {
                    "name": pick["name"],
                    "config": {
                        "strategy": cfg.strategy,
                        "hybrid_vol": cfg.hybrid_vol,
                        "min_concentration_ratio": cfg.min_concentration_ratio,
                        "lieflow_weight": cfg.lieflow_weight,
                        "signal_smoothing": cfg.signal_smoothing,
                        "cost_bps": cfg.cost_bps,
                        "signal_sign": cfg.signal_sign,
                        "min_exposure": cfg.min_exposure,
                        "max_exposure": cfg.max_exposure,
                    },
                    "meets_target": pick["meets_target"],
                    "min_sharpe": pick["min_sharpe"],
                    "mean_total_return": pick["mean_total_return"],
                    "metrics": pick["metrics"],
                },
                indent=2,
            )
        )

    # Pareto frontier among LieFlow configs
    rows = []
    for name, cfg, m in zip(names, configs, metrics_list):
        rows.append(
            {
                "name": name,
                "min_sharpe": float(m["min_sharpe"]),
                "mean_total_return": mean_total_return(m),
            }
        )

    summary = out_dir / "evals-summary.md"
    lines = [
        f"# LieFlow Target Autoresearch\n",
        f"- **Target:** min Sharpe > {args.sharpe_min}, mean PnL > {args.pnl_min:.0%}\n",
        f"- **Configs tested:** {len(configs)}\n",
        f"- **Winners:** {len(winners)}\n",
    ]
    if winners:
        lines.append("\n## Winners (meet both constraints)\n\n")
        lines.append("| Config | Min Sharpe | Mean Return |\n|---|---|---|\n")
        for w in winners[:10]:
            lines.append(
                f"| {w['name']} | {w['min_sharpe']:.3f} | {w['mean_total_return']:.1%} |\n"
            )
        lines.append(f"\n**Best:** `{pick['name']}`\n")
    elif best_row:
        lines.append(
            f"\n## No config met both targets\n\n"
            f"**Closest:** `{best_row['name']}` — "
            f"min Sharpe {best_row['min_sharpe']:.3f}, mean return {best_row['mean_total_return']:.1%} "
            f"(gap {best_row['gap']:.3f})\n"
        )
        # Show near-misses
        near = sorted(rows, key=lambda r: distance_to_target(
            {"min_sharpe": r["min_sharpe"], "periods": {f"p{i}": {"total_return": r["mean_total_return"], "n_days": 1} for i in range(6)}},
            sharpe_min=args.sharpe_min,
            pnl_min=args.pnl_min,
        ))[:8]
        lines.append("\n## Near-miss Pareto configs\n\n")
        lines.append("| Config | Min Sharpe | Mean Return |\n|---|---|---|\n")
        for r in sorted(rows, key=lambda x: (-x["mean_total_return"]))[:5]:
            if r["min_sharpe"] > 0.3:
                lines.append(f"| {r['name']} | {r['min_sharpe']:.3f} | {r['mean_total_return']:.1%} |\n")

    summary.write_text("".join(lines))

    handoff = {
        "version": "2.1.0",
        "source": "loop-lieflow-target",
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "results_tsv": str(tsv),
        "commit": sha,
        "findings": [
            f"n_winners={len(winners)}",
            f"best={pick['name'] if pick else 'none'}",
            f"best_min_sharpe={pick['min_sharpe']:.4f}" if pick else "no_result",
            f"best_mean_return={pick['mean_total_return']:.4f}" if pick else "no_result",
            f"meets_target={pick['meets_target']}" if pick else False,
        ],
        "config": {
            "goal": "LieFlow in loop with min_sharpe>0.5 and mean_pnl>18%",
            "metric": "min_sharpe_and_mean_return",
            "sharpe_min": args.sharpe_min,
            "pnl_min": args.pnl_min,
        },
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))

    if winners:
        print(f"\nCONVERGED: {len(winners)} config(s) meet target")
        for w in winners[:5]:
            print(
                f"  {w['name']:40s} minS={w['min_sharpe']:.3f} "
                f"meanRet={w['mean_total_return']:.1%}"
            )
    else:
        print("\nBOUNDED: no config met both targets")
        if best_row:
            print(
                f"  Closest: {best_row['name']} minS={best_row['min_sharpe']:.3f} "
                f"meanRet={best_row['mean_total_return']:.1%}"
            )

    print(f"Saved -> {BEST_PATH}")
    print(json.dumps(handoff, indent=2))


if __name__ == "__main__":
    main()
