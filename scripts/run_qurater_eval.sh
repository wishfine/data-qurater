#!/bin/bash
# run_qurater_eval.sh
# Run evaluation on validation/test set, computing AUC, BCE, swap consistency, and domain accuracy.

set -e

MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
CHECKPOINT_PATH=${2:-"./outputs/qwen_qurater_full/final_qurater/model.pt"}
EVAL_DATA=${3:-"example_train_data.jsonl"}
OUTPUT_METRICS="./outputs/qwen_qurater_full/eval_metrics.json"

echo "=== STARTING EVALUATION ==="
echo "Model Path      : $MODEL_PATH"
echo "Checkpoint Path : $CHECKPOINT_PATH"
echo "Evaluation Data : $EVAL_DATA"
echo "Output Metrics  : $OUTPUT_METRICS"

python3 evaluate_qurater.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --eval_file "$EVAL_DATA" \
    --max_length 512 \
    --batch_size 4 \
    --output_file "$OUTPUT_METRICS" \
    --pooling_type last_token \
    --head_type A

echo "=== EVALUATION COMPLETED ==="
