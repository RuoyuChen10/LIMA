#!/usr/bin/env bash

# Lambda sensitivity study on 100 correctly predicted ImageNet samples per model.
#
# Paper-facing lambda order used throughout this script:
#   lambda1 = consistency, lambda2 = collaboration,
#   lambda3 = confidence,  lambda4 = effectiveness.
#
# Examples:
#   bash scripts/lambda_ablation.sh
#   GPUS=0,1,2,3 bash scripts/lambda_ablation.sh
#   MODELS=clip,imagebind GPUS=0,1 bash scripts/lambda_ablation.sh
#   RUN_CONFIGS=baseline,lambda1_40 bash scripts/lambda_ablation.sh
#   DRY_RUN=1 bash scripts/lambda_ablation.sh

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-lima}"
SKIP_CONDA="${SKIP_CONDA:-0}"
DATASET="${DATASET:-datasets/imagenet/ILSVRC2012_img_val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-lambda_ablation}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
GPUS="${GPUS:-0}"
MODELS="${MODELS:-clip,imagebind,vit,mamba}"
RUN_CONFIGS="${RUN_CONFIGS:-all}"
SUPERPIXEL_ALGORITHM="${SUPERPIXEL_ALGORITHM:-slico}"
PENDING_SAMPLES="${PENDING_SAMPLES:-8}"
IMAGEBIND_REGION_SIZE="${IMAGEBIND_REGION_SIZE:-30}"
VIT_CHECKPOINT="${VIT_CHECKPOINT:-ckpt/pytorch_model/vit_large_patch16_224_pretrained.pth}"
REFRESH_EVAL_LISTS="${REFRESH_EVAL_LISTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Baseline: (20, 5, 0.05, 0.01), as stated in the review comment.
# One factor is changed at a time. Besides removal/half/double settings, the
# lambda1/lambda2 sweeps include equal-weight endpoints to test lambda1 > lambda2.
ABLATION_CONFIGS=(
    "baseline|20|5|0.05|0.01"
    "lambda1_0|0|5|0.05|0.01"
    "lambda1_5|5|5|0.05|0.01"
    "lambda1_10|10|5|0.05|0.01"
    "lambda1_40|40|5|0.05|0.01"
    "lambda2_0|20|0|0.05|0.01"
    "lambda2_2.5|20|2.5|0.05|0.01"
    "lambda2_10|20|10|0.05|0.01"
    "lambda2_20|20|20|0.05|0.01"
    "lambda3_0|20|5|0|0.01"
    "lambda3_0.025|20|5|0.025|0.01"
    "lambda3_0.1|20|5|0.1|0.01"
    "lambda4_0|20|5|0.05|0"
    "lambda4_0.005|20|5|0.05|0.005"
    "lambda4_0.02|20|5|0.05|0.02"
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

activate_conda() {
    if [[ "$SKIP_CONDA" == "1" ]]; then
        echo "Skipping conda activation (SKIP_CONDA=1)."
        return
    fi

    if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
        echo "Conda environment '$CONDA_ENV' is already active."
        return
    fi

    if ! command -v conda >/dev/null 2>&1; then
        die "conda is not available. Activate '$CONDA_ENV' first, or run with SKIP_CONDA=1."
    fi

    # conda is often an executable rather than a shell function in batch jobs.
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "Activated conda environment '$CONDA_ENV'."
}

contains_csv_value() {
    local csv="$1"
    local wanted="$2"
    local value
    local -a values=()

    [[ "$csv" == "all" ]] && return 0
    IFS=',' read -r -a values <<< "$csv"
    for value in "${values[@]}"; do
        [[ "$value" == "$wanted" ]] && return 0
    done
    return 1
}

model_spec() {
    local model="$1"
    case "$model" in
        clip)
            MODEL_MODULE="submodular_attribution.efficientv2-smdl_explanation_imagenet_clip_vitl_superpixel"
            SOURCE_EVAL_LIST="datasets/imagenet/val_clip_vitl_5k_true.txt"
            MODEL_KIND="multimodal"
            EXTRA_MODEL_ARGS=()
            ;;
        imagebind)
            MODEL_MODULE="submodular_attribution.efficientv2-smdl_explanation_imagenet_imagebind_superpixel"
            SOURCE_EVAL_LIST="datasets/imagenet/val_imagebind_5k_true.txt"
            MODEL_KIND="multimodal"
            EXTRA_MODEL_ARGS=(--region-size "$IMAGEBIND_REGION_SIZE")
            ;;
        vit)
            MODEL_MODULE="submodular_attribution.efficientv2-smdl_explanation_imagenet_vitl_superpixel"
            SOURCE_EVAL_LIST="datasets/imagenet/val_vitl_5k_true.txt"
            MODEL_KIND="single_modal"
            EXTRA_MODEL_ARGS=(--checkpoint "$VIT_CHECKPOINT")
            ;;
        mamba)
            MODEL_MODULE="submodular_attribution.efficientv2-smdl_explanation_imagenet_mambavision_superpixel"
            SOURCE_EVAL_LIST="datasets/imagenet/val_mambavision_5k_true.txt"
            MODEL_KIND="single_modal"
            EXTRA_MODEL_ARGS=()
            ;;
        *)
            die "unknown model '$model' (choose from clip,imagebind,vit,mamba)"
            ;;
    esac
}

prepare_eval_list() {
    local model="$1"
    local target="$OUTPUT_ROOT/eval_lists/${model}_${NUM_SAMPLES}.txt"
    local count

    [[ -f "$SOURCE_EVAL_LIST" ]] || die "missing evaluation list: $SOURCE_EVAL_LIST"
    mkdir -p "$(dirname "$target")"

    # Keep an existing manifest by default: every lambda configuration and every
    # resumed invocation then uses exactly the same samples for this model.
    if [[ ! -f "$target" || "$REFRESH_EVAL_LISTS" == "1" ]]; then
        awk 'NF { print; n++ } n == limit { exit }' limit="$NUM_SAMPLES" \
            "$SOURCE_EVAL_LIST" > "$target"
    fi

    count="$(awk 'NF { n++ } END { print n + 0 }' "$target")"
    [[ "$count" -eq "$NUM_SAMPLES" ]] || \
        die "$target contains $count non-empty rows; expected $NUM_SAMPLES"

    EVAL_LIST="$target"
}

prepare_multimodal_assets() {
    local model="$1"
    local semantic_path=""
    local model_path=""
    local warmup_root="$OUTPUT_ROOT/_asset_warmup/$model"
    local warmup_log="$warmup_root/warmup.log"
    local -a warmup_args=()

    [[ "$MODEL_KIND" == "multimodal" ]] || return 0

    case "$model" in
        clip)
            semantic_path="ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
            model_path=".checkpoints/CLIP/ViT-L-14.pt"
            ;;
        imagebind)
            semantic_path="ckpt/semantic_features/imagebind_imagenet_zeroweights.pt"
            model_path=".checkpoints/imagebind_huge.pth"
            warmup_args=(--region-size "$IMAGEBIND_REGION_SIZE")
            ;;
    esac

    if [[ -f "$semantic_path" && -f "$model_path" ]]; then
        return
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$model] would prepare missing model/semantic assets on GPU ${CUDA_DEVICES[0]}."
        return
    fi

    # CLIP/ImageBind entry points can download a missing backbone and create
    # zero-shot semantic weights. Do this once before multi-GPU launch to avoid
    # several workers writing the same checkpoint concurrently. begin=end=0
    # initializes the model without consuming an evaluation sample.
    mkdir -p "$warmup_root"
    echo "[$model] preparing missing model/semantic assets on GPU ${CUDA_DEVICES[0]} (see $warmup_log)."
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[0]}" python -m "$MODEL_MODULE" \
        --Datasets "$DATASET" \
        --eval-list "$EVAL_LIST" \
        --superpixel-algorithm "$SUPERPIXEL_ALGORITHM" \
        --pending-samples "$PENDING_SAMPLES" \
        --save-dir "$warmup_root" \
        --begin 0 \
        --end 0 \
        --lambda1 0.05 \
        --lambda2 0.01 \
        --lambda3 20 \
        --lambda4 5 \
        "${warmup_args[@]}" \
        > "$warmup_log" 2>&1 || \
        die "asset preparation failed for $model; inspect $warmup_log"

    [[ -f "$semantic_path" ]] || die "$model semantic weights were not created: $semantic_path"
    [[ -f "$model_path" ]] || die "$model checkpoint was not created: $model_path"
}

discover_explanation_dir() {
    local run_root="$1"
    local -a candidates=()
    local path

    [[ -d "$run_root" ]] || return 1
    while IFS= read -r path; do
        [[ -d "$path/npy" && -d "$path/json" ]] && candidates+=("$path")
    done < <(find "$run_root" -mindepth 1 -maxdepth 1 -type d -print | sort)

    if [[ "${#candidates[@]}" -eq 1 ]]; then
        printf '%s\n' "${candidates[0]}"
        return 0
    fi
    if [[ "${#candidates[@]}" -gt 1 ]]; then
        die "multiple backend output directories found under $run_root"
    fi
    return 1
}

count_outputs() {
    local explanation_dir="$1"
    local extension="$2"
    find "$explanation_dir/$extension" -type f -name "*.$extension" -print 2>/dev/null | wc -l
}

run_generation() {
    local model="$1"
    local config_name="$2"
    local semantic_l1="$3"
    local semantic_l2="$4"
    local semantic_l3="$5"
    local semantic_l4="$6"
    local run_root="$OUTPUT_ROOT/$model/$config_name"
    local log_dir="$run_root/logs"
    local explanation_dir=""
    local json_count=0
    local npy_count=0
    local raw_l1 raw_l2 raw_l3 raw_l4
    local gpu_count chunk_size gpu_index begin end device
    local failed=0
    local -a pids=()
    local -a command_args=()

    mkdir -p "$log_dir"

    # The historical multimodal CLI uses the internal order
    # confidence/effectiveness/consistency/collaboration. Reorder paper-facing
    # lambdas here. The single-modal CLI already uses the paper-facing order and
    # has no effectiveness term.
    if [[ "$MODEL_KIND" == "multimodal" ]]; then
        raw_l1="$semantic_l3"
        raw_l2="$semantic_l4"
        raw_l3="$semantic_l1"
        raw_l4="$semantic_l2"
        command_args=(
            --lambda1 "$raw_l1"
            --lambda2 "$raw_l2"
            --lambda3 "$raw_l3"
            --lambda4 "$raw_l4"
        )
    else
        raw_l1="$semantic_l1"
        raw_l2="$semantic_l2"
        raw_l3="$semantic_l3"
        raw_l4="NA"
        command_args=(
            --lambda1 "$raw_l1"
            --lambda2 "$raw_l2"
            --lambda3 "$raw_l3"
        )
    fi

    if explanation_dir="$(discover_explanation_dir "$run_root")"; then
        json_count="$(count_outputs "$explanation_dir" json)"
        npy_count="$(count_outputs "$explanation_dir" npy)"
    fi

    if [[ "$json_count" -eq "$NUM_SAMPLES" && "$npy_count" -eq "$NUM_SAMPLES" ]]; then
        echo "[$model/$config_name] found $NUM_SAMPLES completed samples; generation skipped."
        EXPLANATION_DIR="$explanation_dir"
        return
    fi
    if [[ "$json_count" -ne "$npy_count" ]]; then
        die "incomplete output pair in $run_root (json=$json_count, npy=$npy_count)"
    fi
    if [[ "$json_count" -gt "$NUM_SAMPLES" ]]; then
        die "$run_root contains $json_count samples, more than requested $NUM_SAMPLES"
    fi

    gpu_count="${#CUDA_DEVICES[@]}"
    chunk_size=$(( (NUM_SAMPLES + gpu_count - 1) / gpu_count ))
    echo "[$model/$config_name] semantic lambdas=($semantic_l1,$semantic_l2,$semantic_l3,$semantic_l4), GPUs=${CUDA_DEVICES[*]}"

    for gpu_index in "${!CUDA_DEVICES[@]}"; do
        begin=$((gpu_index * chunk_size))
        (( begin >= NUM_SAMPLES )) && continue
        end=$((begin + chunk_size))
        (( end > NUM_SAMPLES )) && end="$NUM_SAMPLES"
        device="${CUDA_DEVICES[$gpu_index]}"

        local -a cmd=(
            python -m "$MODEL_MODULE"
            --Datasets "$DATASET"
            --eval-list "$EVAL_LIST"
            --superpixel-algorithm "$SUPERPIXEL_ALGORITHM"
            --pending-samples "$PENDING_SAMPLES"
            --save-dir "$run_root"
            --begin "$begin"
            --end "$end"
            "${command_args[@]}"
            "${EXTRA_MODEL_ARGS[@]}"
        )

        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'CUDA_VISIBLE_DEVICES=%q ' "$device"
            printf '%q ' "${cmd[@]}"
            printf '> %q 2>&1 &\n' "$log_dir/gpu_${device}_${begin}_${end}.log"
        else
            CUDA_VISIBLE_DEVICES="$device" "${cmd[@]}" \
                > "$log_dir/gpu_${device}_${begin}_${end}.log" 2>&1 &
            pids+=("$!")
        fi
    done

    if [[ "$DRY_RUN" == "1" ]]; then
        EXPLANATION_DIR="$run_root/<backend-generated-directory>"
        return
    fi

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    [[ "$failed" -eq 0 ]] || die "generation failed for $model/$config_name; inspect $log_dir"

    explanation_dir="$(discover_explanation_dir "$run_root")" || \
        die "no generated explanation directory found under $run_root"
    json_count="$(count_outputs "$explanation_dir" json)"
    npy_count="$(count_outputs "$explanation_dir" npy)"
    [[ "$json_count" -eq "$NUM_SAMPLES" && "$npy_count" -eq "$NUM_SAMPLES" ]] || \
        die "$model/$config_name produced json=$json_count and npy=$npy_count; expected $NUM_SAMPLES each"

    EXPLANATION_DIR="$explanation_dir"
}

evaluate_generation() {
    local model="$1"
    local config_name="$2"
    local semantic_l1="$3"
    local semantic_l2="$4"
    local semantic_l3="$5"
    local semantic_l4="$6"
    local metrics_log="$OUTPUT_ROOT/$model/$config_name/metrics.txt"
    local output insertion deletion

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'python -m evals.eval_AUC_faithfulness --explanation-dir %q\n' "$EXPLANATION_DIR"
        return
    fi

    output="$(python -m evals.eval_AUC_faithfulness \
        --explanation-dir "$EXPLANATION_DIR" 2>&1 | tee "$metrics_log")"
    insertion="$(awk -F': ' '/Insertion AUC Score:/ { print $2 }' <<< "$output" | tail -n 1)"
    deletion="$(awk -F': ' '/Deletion AUC Score:/ { print $2 }' <<< "$output" | tail -n 1)"
    [[ -n "$insertion" && -n "$deletion" ]] || \
        die "could not parse insertion/deletion from $metrics_log"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$model" "$config_name" "$semantic_l1" "$semantic_l2" \
        "$semantic_l3" "$semantic_l4" "$NUM_SAMPLES" "$insertion" "$deletion" \
        >> "$SUMMARY_FILE"
    echo "[$model/$config_name] insertion=$insertion deletion=$deletion"
}

[[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "NUM_SAMPLES must be a positive integer"
[[ "$SUPERPIXEL_ALGORITHM" == "slico" || "$SUPERPIXEL_ALGORITHM" == "seeds" ]] || \
    die "SUPERPIXEL_ALGORITHM must be slico or seeds"

IFS=',' read -r -a CUDA_DEVICES <<< "$GPUS"
[[ "${#CUDA_DEVICES[@]}" -gt 0 ]] || die "GPUS must contain at least one CUDA device"
for device in "${CUDA_DEVICES[@]}"; do
    [[ -n "$device" ]] || die "GPUS contains an empty CUDA device"
done
[[ "${#CUDA_DEVICES[@]}" -le "$NUM_SAMPLES" ]] || \
    die "the number of GPUs cannot exceed NUM_SAMPLES"

IFS=',' read -r -a REQUESTED_MODELS <<< "$MODELS"
[[ "${#REQUESTED_MODELS[@]}" -gt 0 ]] || die "MODELS must contain at least one model"

activate_conda
mkdir -p "$OUTPUT_ROOT"

if [[ "$DRY_RUN" != "1" ]]; then
    [[ -d "$DATASET" ]] || die "missing ImageNet validation directory: $DATASET"
    for model in "${REQUESTED_MODELS[@]}"; do
        if [[ "$model" == "vit" && ! -f "$VIT_CHECKPOINT" ]]; then
            die "missing ViT checkpoint: $VIT_CHECKPOINT (override with VIT_CHECKPOINT=/path/to/checkpoint)"
        fi
    done
fi

SUMMARY_FILE="$OUTPUT_ROOT/metrics.tsv"
if [[ "$DRY_RUN" != "1" ]]; then
    printf 'model\tconfig\tlambda1_consistency\tlambda2_collaboration\tlambda3_confidence\tlambda4_effectiveness\tn_samples\tinsertion_auc\tdeletion_auc\n' \
        > "$SUMMARY_FILE"
fi

for model in "${REQUESTED_MODELS[@]}"; do
    model_spec "$model"
    prepare_eval_list "$model"
    prepare_multimodal_assets "$model"

    for config in "${ABLATION_CONFIGS[@]}"; do
        IFS='|' read -r config_name semantic_l1 semantic_l2 semantic_l3 semantic_l4 <<< "$config"
        contains_csv_value "$RUN_CONFIGS" "$config_name" || continue

        # ViT and MambaVision use the single-modal objective, which has no
        # effectiveness/lambda4 term. Do not launch duplicate no-op runs.
        if [[ "$MODEL_KIND" == "single_modal" && "$config_name" == lambda4_* ]]; then
            continue
        fi

        run_generation "$model" "$config_name" \
            "$semantic_l1" "$semantic_l2" "$semantic_l3" "$semantic_l4"
        evaluate_generation "$model" "$config_name" \
            "$semantic_l1" "$semantic_l2" "$semantic_l3" "$semantic_l4"
    done
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run completed; no experiments were launched."
else
    echo "All requested experiments completed. Summary: $SUMMARY_FILE"
fi
