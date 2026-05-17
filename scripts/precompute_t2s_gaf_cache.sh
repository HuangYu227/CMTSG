#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets/t2s}"
GAF_MAX_SIZE="${GAF_MAX_SIZE:-384}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
OVERWRITE_FLAG=()
if [[ "${OVERWRITE_GAF:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

DATASETS=(traffic airquality ettm1)
HORIZONS=(24 48 96)

for dataset in "${DATASETS[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    name="${dataset}_${horizon}"
    echo "=== Precompute GADF cache: ${name} ==="
    python -m cmtsg.preprocess.precompute_gaf \
      --data-root "${DATA_ROOT}/${name}" \
      --splits train valid test \
      --max-size "${GAF_MAX_SIZE}" \
      --chunk-size "${CHUNK_SIZE}" \
      "${OVERWRITE_FLAG[@]}"
  done
done
