#!/usr/bin/env bash
# server_run_baseline_eval.sh
# Save and evaluate untrained checkpoint-0 baseline model.

set -euo pipefail

# Environment Pre-check via python helper
python scripts/check_environment_status.py

MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
EVAL_DATA="data/qurating/smoke_eval.jsonl"

mkdir -p reports/server outputs/qwen35_4b_experiment/evaluations
OUTPUT_FILE="reports/server/baseline_eval_output.txt"

echo "=== RUNNING BASELINE (CHECKPOINT-0) PREPARATION & EVALUATION ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$OUTPUT_FILE"
echo "----------------------------------------------------------------" | tee -a "$OUTPUT_FILE"

# 1. Save untrained baseline weights and config
python scripts/server_save_baseline.py \
    --model_path "$MODEL_PATH" \
    --validation_file "$EVAL_DATA" 2>&1 | tee -a "$OUTPUT_FILE"

# 2. Evaluate checkpoint-0
python evaluate_qurater.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "outputs/qwen35_4b_experiment/checkpoint-0" \
    --eval_file "$EVAL_DATA" \
    --max_length 256 \
    --batch_size 2 \
    --output_file "outputs/qwen35_4b_experiment/evaluations/baseline_eval.json" 2>&1 | tee -a "$OUTPUT_FILE"

echo "----------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Baseline preparation & evaluation complete. Saved to $OUTPUT_FILE"
