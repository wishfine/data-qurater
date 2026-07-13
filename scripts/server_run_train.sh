#!/bin/bash
# server_run_train.sh
# Run Qwen3.5-4B BF16 LoRA full training (3 epochs, single GPU).

set -euo pipefail

# Environment Pre-check via python helper
python scripts/check_environment_status.py

MODEL_PATH=${1:-"$(cat outputs/model_path.txt 2>/dev/null || echo 'Qwen/Qwen3.5-4B')"}
TRAIN_FILE=${2:-"data/qurating/train.jsonl"}
VAL_FILE=${3:-"data/qurating/eval.jsonl"}

mkdir -p reports/server outputs/qwen35_4b_experiment/evaluations
OUTPUT_FILE="reports/server/full_train_output.txt"

echo "=== RUNNING QWEN3.5-4B FULL TRAINING ===" | tee "$OUTPUT_FILE"
echo "Timestamp  : $(date)" | tee -a "$OUTPUT_FILE"
echo "Model Path : $MODEL_PATH" | tee -a "$OUTPUT_FILE"
echo "Train File : $TRAIN_FILE" | tee -a "$OUTPUT_FILE"
echo "Val File   : $VAL_FILE" | tee -a "$OUTPUT_FILE"
echo "----------------------------------------" | tee -a "$OUTPUT_FILE"

# Run 3 epochs with batch_size=4, grad_accum=4 on 2 GPUs.
# Evaluates and compares checkpoints automatically every 0.25 epochs.
torchrun --nproc_per_node=2 train_qurater_qwen.py \
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
    --seed 42 2>&1 | tee -a "$OUTPUT_FILE"

echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
echo "=== GENERATING CHECKPOINT COMPARISON TABLE ===" | tee -a "$OUTPUT_FILE"

# Generate comparison table for all 0.5-epoch checkpoints
python compare_checkpoints.py \
    --eval_dir "outputs/qwen35_4b_experiment/evaluations" \
    --output_md "reports/server/training_comparison.md" \
    --output_json "reports/server/training_comparison.json" \
    --learning_curve "outputs/qwen35_4b_experiment/evaluations/learning_curve.csv" 2>&1 | tee -a "$OUTPUT_FILE"

echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Full training and evaluation comparisons complete. Saved to $OUTPUT_FILE"
