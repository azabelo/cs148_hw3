#!/usr/bin/env bash
# Short injection-strategy comparison (projector-only / freeze A).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
VIT="${VIT:-runs/clip_eurosat/best.pt}"
STEPS="${STEPS:-150}"
EVAL_MAX="${EVAL_MAX:-200}"
GROUP="${GROUP:-injection_compare_short}"

for INJ in cls all_patches interleaved; do
  echo "========== injection=${INJ} =========="
  "${PY}" scripts/train_vlm.py \
    --config configs/vlm_clevr.yaml \
    --pretrained-vit "${VIT}" \
    --injection "${INJ}" \
    --mask-mode image_bidir \
    --freeze-config A \
    --num-steps "${STEPS}" \
    --grad-accum 1 \
    --eval-every 0 \
    --eval-max-examples "${EVAL_MAX}" \
    --output-dir "runs/vlm_injection_${INJ}_A_short" \
    --wandb \
    --wandb-project cs148-hw3-vlm \
    --wandb-group "${GROUP}" \
    --wandb-run-name "${INJ}_short"
done

echo "Done. Summaries:"
for INJ in cls all_patches interleaved; do
  echo -n "${INJ}: "
  "${PY}" -c "import json; print(json.load(open('runs/vlm_injection_${INJ}_A_short/metrics.json')))"
done
