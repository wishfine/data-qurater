#!/bin/bash
# server_test_checkpoint.sh
# Run evaluation on smoke checkpoint and execute comparative analysis.

set -euo pipefail

MODEL_PATH=${1:-"Qwen/Qwen3-0.6B"}
CHECKPOINT_DIR="outputs/qwen3_06b_experiment/checkpoints/checkpoint-epoch-1"
EVAL_DATA="data/qurating/smoke_eval.jsonl"

mkdir -p reports/server outputs/qwen3_06b_experiment/evaluations
OUTPUT_FILE="reports/server/checkpoint_output.txt"

echo "=== RUNNING CHECKPOINT EVALUATION & COMPARISON ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "Checkpoint Dir: $CHECKPOINT_DIR" | tee -a "$OUTPUT_FILE"
echo "---------------------------------------------------" | tee -a "$OUTPUT_FILE"

# 1. Evaluate trained smoke checkpoint
python3 evaluate_qurater.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --eval_file "$EVAL_DATA" \
    --max_length 256 \
    --batch_size 2 \
    --output_file "outputs/qwen3_06b_experiment/evaluations/smoke_eval.json" 2>&1 | tee -a "$OUTPUT_FILE"

# 2. Run checkpoint comparison script to compare checkpoint-0 and smoke checkpoints
python3 compare_checkpoints.py \
    --eval_dir "outputs/qwen3_06b_experiment/evaluations" \
    --output_md "reports/server/training_comparison.md" \
    --output_json "reports/server/training_comparison.json" \
    --learning_curve "outputs/qwen3_06b_experiment/evaluations/learning_curve.csv" 2>&1 | tee -a "$OUTPUT_FILE"

echo "---------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Checkpoint test and comparative analysis complete. Saved to $OUTPUT_FILE"
