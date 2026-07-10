#!/bin/bash
# server_run_smoke.sh
# Run a 2-step single-GPU smoke test for QwenQuRater training and benchmark performance.

set -euo pipefail

MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
mkdir -p reports/server
OUTPUT_FILE="reports/server/smoke_output.txt"

echo "=== RUNNING SINGLE-GPU SMOKE TEST ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$OUTPUT_FILE"
echo "-------------------------------------" | tee -a "$OUTPUT_FILE"

# Run 8 samples, batch_size=1, grad_accum=4.
# Total steps = 8.
# Optimizer steps = 8 / 4 = 2 steps.
python3 train_qurater_qwen.py \
    --model_path "$MODEL_PATH" \
    --train_file "data/qurating/smoke_train.jsonl" \
    --validation_file "data/qurating/smoke_eval.jsonl" \
    --output_dir "./outputs/smoke_test" \
    --max_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_train_samples 8 \
    --max_eval_samples 8 \
    --use_lora \
    --use_4bit \
    --gradient_checkpointing \
    --seed 42 2>&1 | tee -a "$OUTPUT_FILE"

echo "-------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Smoke test training run complete. Saved to $OUTPUT_FILE"
