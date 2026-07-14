#!/bin/bash
# run_qurater_smoke.sh
# Execute a fast, low-footprint training dry-run to verify imports, configurations, and Bradley-Terry loss directions.

set -e

# Model path defaults to ModelScope Qwen3.5-4B or first argument
MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
TRAIN_DATA="data/qurating/smoke_train.jsonl"
OUTPUT_DIR="./outputs/smoke_test"

echo "=== RUNNING QWENQURATER SMOKE TEST ==="
echo "Model Path: $MODEL_PATH"
echo "Train Data: $TRAIN_DATA"
echo "Output Dir: $OUTPUT_DIR"

# Run 1 epoch, limiting training to 8 samples, batch size 2, gradient accumulation 1
python3 train_qurater_qwen.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_DATA" \
    --validation_file "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_train_samples 8 \
    --use_lora \
    --seed 42

echo "=== SMOKE TEST RUN COMPLETED ==="
