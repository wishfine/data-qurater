#!/bin/bash
# run_qurater_eval.sh
# Run evaluation on validation/test set, computing AUC, BCE, swap consistency, and domain accuracy.

set -e

MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
CHECKPOINT_DIR=${2:-"./outputs/qwen_qurater_full/final_qurater"}
EVAL_DATA=${3:-"data/qurating/smoke_eval.jsonl"}
OUTPUT_METRICS="./outputs/qwen_qurater_full/eval_metrics.json"

echo "=== STARTING EVALUATION ==="
echo "Model Path      : $MODEL_PATH"
echo "Checkpoint Dir  : $CHECKPOINT_DIR"
echo "Evaluation Data : $EVAL_DATA"
echo "Output Metrics  : $OUTPUT_METRICS"

python3 evaluate_qurater.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --eval_file "$EVAL_DATA" \
    --max_length 512 \
    --batch_size 4 \
    --output_file "$OUTPUT_METRICS"

echo "=== EVALUATION COMPLETED ==="
