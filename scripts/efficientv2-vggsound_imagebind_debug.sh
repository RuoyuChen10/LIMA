#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset="${DATASET:-datasets/vggsound/test}"
eval_list="${EVAL_LIST:-datasets/vggsound/val_imagebind_309_false.txt}"
save_dir="${SAVE_DIR:-submodular_results/vggsound-imagebind-efficientv2-debug}"

cuda_device="${CUDA_DEVICE:-0}"
begin="${BEGIN:-0}"
end="${END:-50}"

partition_method="${PARTITION_METHOD:-patch}"
patch_size="${PATCH_SIZE:-8}"
lambda1="${LAMBDA1:-0.0}"
lambda2="${LAMBDA2:-0.05}"
lambda3="${LAMBDA3:-20}"
lambda4="${LAMBDA4:-5}"
pending_samples="${PENDING_SAMPLES:-12}"
conda_env="${CONDA_ENV:-lima}"

find_conda() {
    local candidate

    if [[ -n "${CONDA_BIN:-}" ]]; then
        [[ -x "$CONDA_BIN" ]] || {
            echo "CONDA_BIN is not executable: $CONDA_BIN" >&2
            return 1
        }
        printf '%s\n' "$CONDA_BIN"
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return
    fi

    for candidate in \
        "${CONDA_EXE:-}" \
        "${HOME:-}/anaconda3/bin/conda" \
        "${HOME:-}/miniconda3/bin/conda" \
        "/opt/conda/bin/conda"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    echo "Conda was not found. Set CONDA_BIN=/absolute/path/to/conda." >&2
    return 1
}

[[ -d "$dataset" ]] || {
    echo "VGGSound dataset directory not found: $dataset" >&2
    exit 1
}
[[ -f "$eval_list" ]] || {
    echo "Evaluation list not found: $eval_list" >&2
    exit 1
}

conda_bin="$(find_conda)"

echo "Running VGGSound/ImageBind debug attribution on GPU ${cuda_device}, samples [${begin}, ${end}), ${partition_method} ${patch_size}x${patch_size}."

CUDA_VISIBLE_DEVICES="$cuda_device" "$conda_bin" run --no-capture-output -n "$conda_env" \
python -m submodular_attribution.vggsound_imagebind \
    --Datasets "$dataset" \
    --eval-list "$eval_list" \
    --partition-method "$partition_method" \
    --grad-partition-size "$patch_size" \
    --lambda1 "$lambda1" \
    --lambda2 "$lambda2" \
    --lambda3 "$lambda3" \
    --lambda4 "$lambda4" \
    --pending-samples "$pending_samples" \
    --record-counterfactual \
    --save-dir "$save_dir" \
    --begin "$begin" \
    --end "$end"
