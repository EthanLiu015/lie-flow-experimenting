#!/usr/bin/env bash
# Run LieFlow experiments from the project root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

LIEFLOW_DIR="$ROOT/vendor/lieflow"
cd "$LIEFLOW_DIR"

# Resolve data paths relative to project root.
export HYDRA_FULL_ERROR=1

case "${1:-help}" in
  baseline-2d)
    python experiments/flow_matching_2d.py \
      dataset=C4_arrow model=flow_matching/SO2_to_C4_arrow \
      device="${DEVICE:-cpu}" "$@"
    ;;
  synthetic-2d)
    python experiments/flow_matching_2d.py \
      dataset=C4_factor_cross_section model=flow_matching/SO2_to_C4_factor \
      device="${DEVICE:-cpu}" "$@"
    ;;
  equity-3d)
    python experiments/flow_matching_3d.py \
      dataset=SO3_equity_cross_section model=flow_matching/SO3_equity_cross_section \
      device="${DEVICE:-cpu}" "$@"
    ;;
  *)
    echo "Usage: $0 {baseline-2d|synthetic-2d|equity-3d} [hydra overrides...]"
    exit 1
    ;;
esac
