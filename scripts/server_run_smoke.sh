#!/bin/bash
# server_run_smoke.sh
# Run Qwen3-0.6B BF16 LoRA smoke test (2 optimizer steps, single GPU, no 4-bit).

set -euo pipefail

# Environment Pre-check
if [ ! -f "reports/server/environment_status.json" ] || [ "$(grep -o '"status": *"[^"]*"' reports/server/environment_status.json | cut -d'"' -f4)" != "PASS" ]; then
    echo "Environment verification has not passed."
    echo "Run: bash scripts/server_check_env.sh"
    exit 1
fi

MODEL_PATH=${1:-"Qwen/Qwen3-0.6B"}
mkdir -p reports/server
OUTPUT_FILE="reports/server/smoke_output.txt"

echo "=== RUNNING QWEN3-0.6B SMOKE TRAINING TEST ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$OUTPUT_FILE"
echo "----------------------------------------------" | tee -a "$OUTPUT_FILE"

# Run 8 training pairs, batch_size=1, grad_accum=4.
# Total micro-steps = 8.
# Stop training exactly after 2 optimizer steps.
python3 train_qurater_qwen.py \
    --model_path "$MODEL_PATH" \
    --train_file "data/qurating/smoke_train.jsonl" \
    --validation_file "data/qurating/smoke_eval.jsonl" \
    --output_dir "outputs/qwen3_06b_experiment/checkpoints" \
    --max_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_train_samples 8 \
    --max_eval_samples 8 \
    --max_optimizer_steps 2 \
    --bf16 \
    --confidence_threshold 0.5 \
    --seed 42 2>&1 | tee -a "$OUTPUT_FILE"

echo "----------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Smoke test training complete. Saved to $OUTPUT_FILE"
