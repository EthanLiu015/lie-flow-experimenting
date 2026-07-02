#!/usr/bin/env python3
"""Focused LieFlow sweep: min Sharpe > 0.5 AND mean PnL > 18%."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

from lieflow_quant.session import EvalSession, MultiWindowConfig

ROOT = Path(__file__).resolve().parents[1]
SHARPE_MIN, PNL_MIN = 0.5, 0.18


def mean_ret(m: dict) -> float:
    rets = [float(p["total_return"]) for p in m["periods"].values() if p.get("n_days", 0) > 0]
    return sum(rets) / len(rets) if rets else float("nan")


def build_focused() -> list[tuple[str, MultiWindowConfig]]:
    exps: list[tuple[str, MultiWindowConfig]] = []
    for conc in [0.99, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05]:
        for smooth in [12, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            for lw in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for cost in [5.0, 10.0]:
                    exps.append(
                        (
                            f"hybrid c{conc:.3f} s{smooth} w{lw} cost{int(cost)}",
                            MultiWindowConfig(
                                hybrid_vol=True,
                                min_concentration_ratio=conc,
                                signal_smoothing=smooth,
                                lieflow_weight=lw,
                                cost_bps=cost,
                            ),
                        )
                    )
    for conc in [1.0, 1.02, 1.03, 1.04]:
        for smooth in [16, 18, 20]:
            for min_e, max_e in [(0.6, 1.0), (0.75, 1.0), (0.5, 1.25)]:
                exps.append(
                    (
                        f"hybrid c{conc} s{smooth} exp{min_e}-{max_e}",
                        MultiWindowConfig(
                            hybrid_vol=True,
                            min_concentration_ratio=conc,
                            signal_smoothing=smooth,
                            lieflow_weight=0.5,
                            cost_bps=5.0,
                            min_exposure=min_e,
                            max_exposure=max_e,
                        ),
                    )
                )
    return exps


def main() -> None:
    workers = int(os.environ.get("AR_WORKERS", str(os.cpu_count() or 1)))
    print("[autoresearch] mode: classic")
    print(f"[autoresearch] metric: lieflow_in_loop | Verify: min_sharpe>{SHARPE_MIN} AND mean_return>{PNL_MIN:.0%}")

    exps = build_focused()
    session = EvalSession(
        data_dir=ROOT / "data/equity",
        cache_path=ROOT / "outputs/strategy/inference_cache_full_n60.npz",
        n_target=60,
    )
    print(f"configs={len(exps)} workers={workers}")

    t0 = perf_counter()
    metrics = session.sweep_multiwindow([c for _, c in exps], n_workers=workers)
    print(f"sweep_ms={(perf_counter() - t0) * 1000:.0f}")

    winners: list[dict] = []
    best: dict | None = None
    best_gap = float("inf")
    rows: list[dict] = []

    for (name, _), m in zip(exps, metrics):
        mr = mean_ret(m)
        ms = float(m["min_sharpe"])
        gap = max(0.0, SHARPE_MIN - ms) + max(0.0, PNL_MIN - mr)
        row = {"name": name, "min_sharpe": ms, "mean_ret": mr, "gap": gap, "metrics": m}
        rows.append(row)
        if ms > SHARPE_MIN and mr > PNL_MIN:
            winners.append(row)
        if gap < best_gap:
            best_gap = gap
            best = row

    winners.sort(key=lambda r: (-r["mean_ret"], -r["min_sharpe"]))
    out_dir = ROOT / "autoresearch" / f"loop-lieflow-target-{datetime.now().strftime('%y%m%d-%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"
    with open(tsv, "w") as f:
        f.write("name\tmin_sharpe\tmean_return\tgap\tmeets_target\n")
        for r in rows:
            ok = r["min_sharpe"] > SHARPE_MIN and r["mean_ret"] > PNL_MIN
            f.write(f"{r['name']}\t{r['min_sharpe']:.4f}\t{r['mean_ret']:.4f}\t{r['gap']:.4f}\t{ok}\n")

    status = "CONVERGED" if winners else "BOUNDED"
    pick = winners[0] if winners else best
    if pick:
        (ROOT / "outputs/strategy/best_lieflow_target_config.json").write_text(
            json.dumps({"name": pick["name"], "meets_target": bool(winners), **pick}, indent=2, default=str)
        )

    print(f"status={status} winners={len(winners)}")
    for w in winners[:10]:
        print(f"  {w['name']} minS={w['min_sharpe']:.3f} ret={w['mean_ret']:.1%}")
    if not winners and best:
        print(f"  closest: {best['name']} minS={best['min_sharpe']:.3f} ret={best['mean_ret']:.1%} gap={best['gap']:.3f}")

    near = sorted([r for r in rows if r["min_sharpe"] > 0.45], key=lambda r: -r["mean_ret"])[:8]
    print("top return with minS>0.45:")
    for r in near:
        print(f"  {r['name']} ret={r['mean_ret']:.1%} minS={r['min_sharpe']:.3f}")

    handoff = {
        "status": status,
        "n_winners": len(winners),
        "best": pick["name"] if pick else None,
        "results_tsv": str(tsv),
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2))
    print(json.dumps(handoff, indent=2))


if __name__ == "__main__":
    main()
