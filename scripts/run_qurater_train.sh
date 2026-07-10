#!/bin/bash
# run_qurater_train.sh
# Main script to run full training of the QwenQuRater pairwise model.

set -e

# Model and dataset paths
MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
TRAIN_DATA=${2:-"data/qurating/smoke_train.jsonl"}
VAL_DATA=${3:-"data/qurating/smoke_eval.jsonl"}
OUTPUT_DIR="./outputs/qwen_qurater_full"

echo "=== STARTING FULL QWENQURATER PAIRWISE TRAINING ==="
echo "Model Path       : $MODEL_PATH"
echo "Training Data    : $TRAIN_DATA"
echo "Validation Data  : $VAL_DATA"
echo "Output Directory : $OUTPUT_DIR"

# Run full training using QLoRA (4-bit NF4) to fit comfortably on 80GB PCIe or smaller GPUs
python3 train_qurater_qwen.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_DATA" \
    --validation_file "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --use_lora \
    --use_4bit \
    --gradient_checkpointing \
    --seed 42

echo "=== FULL TRAINING COMPLETED ==="
