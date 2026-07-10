#!/bin/bash
# server_verify_data.sh
# Run pairwise data format, direction verification, and train-eval leakage checks.

set -euo pipefail

# Environment Pre-check via python helper
python3 scripts/check_environment_status.py

mkdir -p reports/server
OUTPUT_FILE="reports/server/data_direction_output.txt"

echo "=== RUNNING DATA FORMAT, DIRECTION, AND LEAKAGE VERIFICATION ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "---------------------------------------------------------------" | tee -a "$OUTPUT_FILE"

# 1. Execute data direction checks
python3 scripts/verify_data_direction.py 2>&1 | tee -a "$OUTPUT_FILE"

# 2. Execute train-eval overlap & leakage checks
echo "---------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Checking train/eval split overlaps and leakage..." | tee -a "$OUTPUT_FILE"
python3 scripts/check_train_eval_overlap.py \
    --train_file "data/qurating/smoke_train.jsonl" \
    --eval_file "data/qurating/smoke_eval.jsonl" \
    --output_report "reports/server/train_eval_overlap_report.json" 2>&1 | tee -a "$OUTPUT_FILE"

echo "---------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Data verification complete. Saved to $OUTPUT_FILE"
