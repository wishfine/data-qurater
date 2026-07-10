#!/bin/bash
# server_test_checkpoint.sh
# Check env status and execute checkpoint round-trip, evaluation, and comparative audits.

set -euo pipefail

# Environment Pre-check via python helper
python scripts/check_environment_status.py

MODEL_PATH=${1:-"Qwen/Qwen3.5-4B"}
CHECKPOINT_DIR="outputs/qwen35_4b_experiment/checkpoints/checkpoint-epoch-1"
EVAL_DATA="data/qurating/smoke_eval.jsonl"

mkdir -p reports/server outputs/qwen35_4b_experiment/evaluations

# 2. Step 1: Run actual checkpoint round-trip verification
echo "=== STEP 1: RUNNING CHECKPOINT ROUND-TRIP VERIFICATION ==="
python scripts/server_checkpoint_roundtrip.py \
    --model_path "$MODEL_PATH" \
    --eval_file "$EVAL_DATA" \
    --roundtrip_tolerance 1e-3 2>&1 | tee reports/server/checkpoint_roundtrip_log.txt

# 3. Step 2: Run evaluation on smoke checkpoint-1
echo "=== STEP 2: RUNNING EVALUATION ON SMOKE CHECKPOINT ==="
python evaluate_qurater.py \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --eval_file "$EVAL_DATA" \
    --max_length 256 \
    --batch_size 2 \
    --output_file "outputs/qwen35_4b_experiment/evaluations/smoke_eval.json" 2>&1 | tee reports/server/smoke_eval_log.txt

# 4. Step 3: Run baseline comparison
echo "=== STEP 3: COMPARING BASELINE AND SMOKE METRICS ==="
python compare_checkpoints.py \
    --eval_dir "outputs/qwen35_4b_experiment/evaluations" \
    --output_md "reports/server/training_comparison.md" \
    --output_json "reports/server/training_comparison.json" \
    --learning_curve "outputs/qwen35_4b_experiment/evaluations/learning_curve.csv" 2>&1 | tee reports/server/checkpoint_comparison_log.txt

echo "[SUCCESS] All checkpoint tests and evaluations completed successfully."
