# LieFlow Quant Symmetry Discovery

Quantitative finance extension of [LieFlow](https://github.com/jypark0/lieflow) ([arxiv:2512.20043](https://arxiv.org/abs/2512.20043)): discovering rotational symmetries in cross-sectional risk feature geometry.

## Research question

Do equity cross-sections in `(momentum, volatility, size)` exhibit approximate rotational symmetries, and does that structure vary across VIX regimes?

LieFlow learns a distribution over a hypothesis Lie group `G`; the support of the learned distribution reveals the true symmetry subgroup `H ⊆ G`. This repo encodes finance data as point clouds and evaluates symmetry recovery on synthetic (ground-truth C4) and real market panels.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e vendor/lieflow
pip install -e .
```

## Project structure

```
lie-flow-experimenting/
├── vendor/lieflow/          # upstream LieFlow (MIT)
├── src/lieflow_quant/       # finance datasets + evaluation
├── scripts/                 # data prep, training wrappers, analysis
├── conf/                    # (configs live in vendor/lieflow/conf)
└── data/                    # generated data (gitignored)
```

## Quick start

### 1. Synthetic C4 factor cross-section (ground truth)

```bash
PYTHONPATH=src python scripts/build_synthetic_factor_cloud.py

cd vendor/lieflow
PYTHONPATH=../../src WANDB_MODE=disabled python experiments/flow_matching_2d.py \
  dataset=C4_factor_cross_section \
  model=flow_matching/SO2_to_C4_factor \
  device=cpu
```

Evaluate recovery:

```bash
PYTHONPATH=src python scripts/evaluate_symmetry_recovery.py \
  --config-dir vendor/lieflow/conf \
  --checkpoint vendor/lieflow/outputs/<date>/<time>/ckpt/model.pt \
  --output-dir outputs/eval_synthetic
```

### 2. Real equity cross-sections

```bash
PYTHONPATH=src python scripts/download_equity_panel.py
PYTHONPATH=src python scripts/build_cross_section_clouds.py

cd vendor/lieflow
PYTHONPATH=../../src WANDB_MODE=disabled python experiments/flow_matching_3d.py \
  dataset=SO3_equity_cross_section \
  model=flow_matching/SO3_equity_cross_section \
  device=cpu
```

Regime analysis (VIX terciles):

```bash
PYTHONPATH=src python scripts/analyze_regime_symmetry.py \
  --config-dir vendor/lieflow/conf \
  --checkpoint vendor/lieflow/outputs/<date>/<time>/ckpt/model.pt \
  --output-dir outputs/regime_analysis
```

### 3. Trading strategy (canonical-residual L/S)

Generate daily signals from LieFlow symmetry inference (out-of-sample last 20% of days by default):

```bash
PYTHONPATH=src python scripts/generate_trading_signals.py \
  --checkpoint vendor/lieflow/outputs/2026-07-01/16-25-52/ckpt/model.pt \
  --output outputs/strategy/daily_signals.csv \
  --device cpu
```

Backtest vs cross-sectional momentum benchmark:

```bash
PYTHONPATH=src python scripts/backtest_strategy.py \
  --signals outputs/strategy/daily_signals.csv \
  --output-dir outputs/strategy
```

**Strategy logic:** rank stocks by canonical-frame momentum residual (long winners, short losers), dollar-neutral portfolio scaled by symmetry concentration (reduce exposure when geometry is unstable). Trades use a 1-day lag on close-to-close returns with 5 bps turnover cost.

## Data encoding

| Experiment | Point cloud | Hypothesis group | Imposed symmetry |
|---|---|---|---|
| Synthetic factor | 20 stocks × (momentum, vol) | SO(2) | C4 (90° rotations) |
| Equity panel | 50 stocks × (mom, vol, log price) | SO(3) | None (inferred) |

## References

- Chen et al., *Discovering Symmetry Groups with Flow Matching*, ICML 2026. [arxiv:2512.20043](https://arxiv.org/abs/2512.20043)
- LieFlow code: [jypark0/lieflow](https://github.com/jypark0/lieflow) (MIT)

## Results (smoke-test runs on CPU)

Minimal training configs were used for local verification:

| Experiment | Checkpoint | Key metric |
|---|---|---|
| Synthetic C4 factor | `vendor/lieflow/outputs/2026-07-01/16-24-44/ckpt/model.pt` | C4 recall 0.75, peaks near 90°/180°/270° — see `outputs/eval_synthetic/` |
| Equity SO(3) | `vendor/lieflow/outputs/2026-07-01/16-25-52/ckpt/model.pt` | Regime histograms in `outputs/regime_analysis/` (high-VIX concentration slightly higher) |

Re-run with more epochs (`train.epochs=300+`) for paper-quality symmetry recovery.

- Training on CPU is slow; reduce `train.epochs` and `dataset.*.num_samples` for quick tests.
- `vendor/lieflow/src/lieflow/utils.py` includes a scipy 1.18 compatibility fix for `logm`.
- Set `WANDB_MODE=disabled` to skip Weights & Biases logging.
