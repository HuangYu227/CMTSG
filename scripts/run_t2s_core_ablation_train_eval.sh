#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
CONFIG_DIR="${CONFIG_DIR:-configs/t2s}"
ABLATION_CONFIG_DIR="${ABLATION_CONFIG_DIR:-configs/t2s_ablation}"
SUMMARY_CSV="${SUMMARY_CSV:-runs/t2s_ablation/summary.csv}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10}"
AGGREGATION="${AGGREGATION:-median}"
FAMILIES="${FAMILIES:-no_grounding no_spectral}"

python scripts/create_t2s_core_ablation_configs.py \
  --config-dir "${CONFIG_DIR}" \
  --output-dir "${ABLATION_CONFIG_DIR}"

for family in ${FAMILIES}; do
  for config in "${ABLATION_CONFIG_DIR}/${family}"/*.yaml; do
    name="$(basename "${config}" .yaml)"
    output_root="runs/t2s_ablation/${family}/${name}"
    echo "=== Train ${family}/${name} ==="
    python -m cmtsg.train \
      --config "${config}" \
      --device "${DEVICE}" \
      --output-root "${output_root}" \
      --sample-every 0

    echo "=== Evaluate ${family}/${name} ==="
    python -m cmtsg.evaluate_t2s_protocol \
      --config "${config}" \
      --checkpoint "${output_root}/checkpoints/best.pt" \
      --device "${DEVICE}" \
      --split test \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --n-samples "${EVAL_SAMPLES}" \
      --aggregation "${AGGREGATION}" \
      --gaf-mode real \
      --summary-csv "${SUMMARY_CSV}" \
      --tag "${family}_${name}_test_full"
  done
done

echo "Core ablation summary: ${SUMMARY_CSV}"
