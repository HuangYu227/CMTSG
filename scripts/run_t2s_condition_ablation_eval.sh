#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: bash scripts/run_t2s_condition_ablation_eval.sh <config.yaml> <checkpoint.pt> [tag_prefix]}"
CHECKPOINT="${2:?Usage: bash scripts/run_t2s_condition_ablation_eval.sh <config.yaml> <checkpoint.pt> [tag_prefix]}"
TAG_PREFIX="${3:-condition_ablation}"

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_SAMPLES="${N_SAMPLES:-10}"
AGGREGATION="${AGGREGATION:-median}"
SUMMARY_CSV="${SUMMARY_CSV:-runs/t2s_condition_ablation/summary.csv}"
GAF_SEED="${GAF_SEED:-123}"

run_eval() {
  local mode="$1"
  local tag="$2"
  python -m cmtsg.evaluate_t2s_protocol \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --split "${SPLIT}" \
    --batch-size "${BATCH_SIZE}" \
    --n-samples "${N_SAMPLES}" \
    --aggregation "${AGGREGATION}" \
    --gaf-mode "${mode}" \
    --gaf-seed "${GAF_SEED}" \
    --summary-csv "${SUMMARY_CSV}" \
    --tag "${TAG_PREFIX}_${tag}"
}

run_eval real full
run_eval none text_only
run_eval random random_gaf
run_eval shuffle shuffled_gaf

echo "Condition ablation summary: ${SUMMARY_CSV}"
