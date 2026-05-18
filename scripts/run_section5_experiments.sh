#!/usr/bin/env bash
# Run §5 masking, freezing, and qualitative experiments on the H100 allocation.
# Usage (from repo root on the compute node):
#   srun --jobid=63726941 --overlap --pty bash
#   bash scripts/run_section5_experiments.sh

set -euo pipefail
cd "$(dirname "$0")/.."

VIT=runs/clip_eurosat/best.pt
CFG=configs/vlm_clevr.yaml
RESULTS=runs/hw3_section5_results.json

if [[ ! -f data/clevr_mini/train.jsonl ]]; then
  echo "Extracting CLEVR-mini..."
  unzip -o data/clevr_mini.zip -d data
fi

train_one() {
  local out="$1"; shift
  echo "=== Training -> ${out} ==="
  uv run python scripts/train_vlm.py \
    --config "${CFG}" \
    --pretrained-vit "${VIT}" \
    "$@" \
    --output-dir "${out}"
}

# --- 5.5 Masking (500 steps, projector-only) ---
train_one runs/vlm_mask_causal_500 \
  --injection all_patches --mask-mode causal --freeze-config A --num-steps 500

train_one runs/vlm_mask_image_bidir_500 \
  --injection all_patches --mask-mode image_bidir --freeze-config A --num-steps 500

# --- 5.6 Freezing (1500 steps, best: all_patches + image_bidir) ---
for fc in A B C D; do
  train_one "runs/vlm_freeze_${fc}_1500" \
    --injection all_patches --mask-mode image_bidir --freeze-config "${fc}" --num-steps 1500
done

# --- 5.7 Qualitative (best checkpoint) ---
BEST=runs/vlm_freeze_C_1500/best.pt
if [[ ! -f "${BEST}" ]]; then
  BEST=runs/vlm_mask_image_bidir_500/best.pt
fi
uv run python scripts/eval_vlm.py \
  --checkpoint "${BEST}" \
  --num-examples 10 --max-eval 500 --save-images \
  --output-dir runs/vlm_qualitative

# Aggregate metrics
uv run python - <<'PY'
import json
from pathlib import Path

runs = {
    "masking_causal_500": Path("runs/vlm_mask_causal_500/metrics.json"),
    "masking_image_bidir_500": Path("runs/vlm_mask_image_bidir_500/metrics.json"),
}
for fc in "ABCD":
    runs[f"freeze_{fc}_1500"] = Path(f"runs/vlm_freeze_{fc}_1500/metrics.json")

out = {}
for name, path in runs.items():
    if path.exists():
        out[name] = json.loads(path.read_text())

Path("runs/hw3_section5_results.json").write_text(json.dumps(out, indent=2))
print("Wrote runs/hw3_section5_results.json")
PY

echo "Done."
