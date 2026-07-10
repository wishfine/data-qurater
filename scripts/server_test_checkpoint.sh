#!/bin/bash
# server_test_checkpoint.sh
# Test loading modular checkpoints and executing evaluation round-trip on the server.

set -euo pipefail

MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
CHECKPOINT_DIR="./outputs/smoke_test/checkpoint-epoch-1"
EVAL_DATA="data/qurating/smoke_eval.jsonl"

mkdir -p reports/server
OUTPUT_FILE="reports/server/checkpoint_output.txt"

echo "=== RUNNING CHECKPOINT LOAD & EVALUATION ROUND-TRIP ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "Checkpoint Dir: $CHECKPOINT_DIR" | tee -a "$OUTPUT_FILE"
echo "--------------------------------------------------------" | tee -a "$OUTPUT_FILE"

# Evaluate using reloaded modular checkpoint
python3 evaluate_qurater.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --eval_file "$EVAL_DATA" \
    --max_length 256 \
    --batch_size 2 2>&1 | tee -a "$OUTPUT_FILE"

echo "--------------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Checkpoint round-trip verification complete. Saved to $OUTPUT_FILE"
