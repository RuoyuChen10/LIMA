#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset="${DATASET:-datasets/medical_lung/lung_dataset}"
eval_list="${EVAL_LIST:-datasets/medical_lung/LC25000_lung_quilt_1k_false.txt}"
save_dir="${SAVE_DIR:-submodular_results/lung-quilt-efficientv2-debug}"

cuda_device="${CUDA_DEVICE:-0}"
begin="${BEGIN:-0}"
end="${END:-5}"

superpixel_algorithm="${SUPERPIXEL_ALGORITHM:-slico}"
lambda1="${LAMBDA1:-0}"
lambda2="${LAMBDA2:-0.05}"
lambda3="${LAMBDA3:-1}"
lambda4="${LAMBDA4:-1}"
pending_samples="${PENDING_SAMPLES:-8}"
conda_env="${CONDA_ENV:-lima}"

echo "Running lung/Quilt counterfactual attribution on GPU ${cuda_device}, samples [${begin}, ${end})."

CUDA_VISIBLE_DEVICES="$cuda_device" conda run --no-capture-output -n "$conda_env" \
python -m submodular_attribution.efficientv2-smdl_explanation_lung_quilt_superpixel \
    --Datasets "$dataset" \
    --eval-list "$eval_list" \
    --superpixel-algorithm "$superpixel_algorithm" \
    --lambda1 "$lambda1" \
    --lambda2 "$lambda2" \
    --lambda3 "$lambda3" \
    --lambda4 "$lambda4" \
    --pending-samples "$pending_samples" \
    --record-counterfactual \
    --save-dir "$save_dir" \
    --begin "$begin" \
    --end "$end"
