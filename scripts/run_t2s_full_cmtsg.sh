#!/usr/bin/env bash
set -euo pipefail

T2S_SOURCE="${T2S_SOURCE:-}"
if [[ -z "${T2S_SOURCE}" ]]; then
  echo "Set T2S_SOURCE to Three Levels Data.zip or the extracted Three Levels Data directory." >&2
  echo "Example: T2S_SOURCE='/data/Three Levels Data.zip' CUDA_VISIBLE_DEVICES=5 bash scripts/run_t2s_full_cmtsg.sh" >&2
  exit 2
fi

DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-datasets/t2s}"
PROCESSED_ROOT="${PROCESSED_ROOT:-processed/t2s}"
SUMMARY_CSV="${SUMMARY_CSV:-runs/t2s_full/summary.csv}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10}"
AGGREGATION="${AGGREGATION:-median}"

if [[ "${RESET_SUMMARY:-0}" == "1" ]]; then
  rm -f "${SUMMARY_CSV}"
fi

python -m cmtsg.preprocess.convert_t2s_three_levels \
  --source "${T2S_SOURCE}" \
  --data-root "${DATA_ROOT}" \
  --processed-root "${PROCESSED_ROOT}" \
  --datasets traffic airquality ettm1 \
  --horizons 24 48 96

CONFIGS=(
  configs/t2s/traffic_24.yaml
  configs/t2s/traffic_48.yaml
  configs/t2s/traffic_96.yaml
  configs/t2s/airquality_24.yaml
  configs/t2s/airquality_48.yaml
  configs/t2s/airquality_96.yaml
  configs/t2s/ettm1_24.yaml
  configs/t2s/ettm1_48.yaml
  configs/t2s/ettm1_96.yaml
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .yaml)"
  output_root="runs/t2s_full/${name}"
  echo "=== Train ${name} ==="
  python -m cmtsg.train \
    --config "${config}" \
    --device "${DEVICE}" \
    --output-root "${output_root}" \
    --sample-every 0

  echo "=== Evaluate ${name} ==="
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
    --tag "${name}_test_full"
done

echo "T2S full CMTSG summary: ${SUMMARY_CSV}"
