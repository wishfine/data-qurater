#!/bin/bash
# server_run_train.sh
# Run Qwen3.5-4B BF16 LoRA full training (3 epochs, single GPU).

set -euo pipefail

# Environment Pre-check via python helper
python scripts/check_environment_status.py
python -c 'import torch; assert torch.cuda.device_count() >= 2, "server_run_train.sh requires at least 2 visible GPUs"'

MODEL_PATH=${1:-"$(cat outputs/model_path.txt 2>/dev/null || echo 'Qwen/Qwen3.5-4B')"}
TRAIN_FILE=${2:-"data/qurating/train.jsonl"}
VAL_FILE=${3:-"data/qurating/eval.jsonl"}
CONFIG_FILE=${4:-"configs/qwen3_06b_train.json"}
RESUME_FROM_CHECKPOINT=${5:-""}

RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
        echo "[ERROR] Resume checkpoint directory does not exist: $RESUME_FROM_CHECKPOINT" >&2
        exit 1
    fi
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p reports/server outputs/qwen35_4b_experiment/evaluations
OUTPUT_FILE="reports/server/full_train_output.txt"

python scripts/check_train_eval_overlap.py \
    --train_file "$TRAIN_FILE" \
    --eval_file "$VAL_FILE" \
    --output_report "reports/server/full_train_overlap_report.json" \
    --skip_manifest_check

echo "=== RUNNING QWEN3.5-4B FULL TRAINING ===" | tee "$OUTPUT_FILE"
echo "Timestamp  : $(date)" | tee -a "$OUTPUT_FILE"
echo "Model Path : $MODEL_PATH" | tee -a "$OUTPUT_FILE"
echo "Train File : $TRAIN_FILE" | tee -a "$OUTPUT_FILE"
echo "Val File   : $VAL_FILE" | tee -a "$OUTPUT_FILE"
echo "Config File: $CONFIG_FILE" | tee -a "$OUTPUT_FILE"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    echo "Resume From: $RESUME_FROM_CHECKPOINT" | tee -a "$OUTPUT_FILE"
else
    echo "Resume From: <fresh run>" | tee -a "$OUTPUT_FILE"
fi
echo "----------------------------------------" | tee -a "$OUTPUT_FILE"

# Run 3 epochs with batch_size=4 and grad_accum=4 on 2 GPUs.
# Checkpoints are saved every 0.25 epoch. Evaluation is intentionally omitted;
# run evaluate_qurater.py separately on an available GPU after training.
torchrun --nproc_per_node=2 train_qurater_qwen.py \
    --config "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --output_dir "outputs/qwen35_4b_experiment/checkpoints" \
    --max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --confidence_threshold 0.5 \
    --max_train_samples 64000 \
    --seed 42 \
    "${RESUME_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_FILE"

echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Training complete. Evaluation was skipped by design; run evaluate_qurater.py separately." | tee -a "$OUTPUT_FILE"
