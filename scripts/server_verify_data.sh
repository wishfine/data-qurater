#!/bin/bash
# server_verify_data.sh
# Automate smoke split regeneration, SHA256/leakage audits, and pairwise direction verification.

set -euo pipefail

# 1. Execute live environment pre-check
python scripts/check_environment_status.py

mkdir -p reports/server
OUTPUT_FILE="reports/server/data_direction_output.txt"

echo "=== RUNNING DATA FORMAT, DIRECTION, AND LEAKAGE VERIFICATION ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "---------------------------------------------------------------" | tee -a "$OUTPUT_FILE"

# 2. Assert source dataset exists
echo "Checking source dataset existence..." | tee -a "$OUTPUT_FILE"
test -f data/qurating/smoke_train_source.jsonl

# 3. Clean old smoke split files to prevent stale state usage
echo "Cleaning old smoke splits..." | tee -a "$OUTPUT_FILE"
rm -f data/qurating/smoke_train.jsonl data/qurating/smoke_eval.jsonl data/qurating/smoke_split_manifest.json

# 4. Rebuild smoke split deterministically (text-disjoint partitioned by connected component)
echo "Regenerating smoke splits from source..." | tee -a "$OUTPUT_FILE"
python scripts/build_smoke_split.py \
  --source_file data/qurating/smoke_train_source.jsonl \
  --train_file data/qurating/smoke_train.jsonl \
  --eval_file data/qurating/smoke_eval.jsonl \
  --manifest_file data/qurating/smoke_split_manifest.json \
  --seed 42 2>&1 | tee -a "$OUTPUT_FILE"

# 5. Assert manifest generated successfully
test -f data/qurating/smoke_split_manifest.json

# 6. Execute train-eval overlap, single-text leakage & manifest SHA256 audits
echo "---------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Auditing train/eval split leakage and manifest SHA256 alignment..." | tee -a "$OUTPUT_FILE"
python scripts/check_train_eval_overlap.py \
  --train_file data/qurating/smoke_train.jsonl \
  --eval_file data/qurating/smoke_eval.jsonl \
  --manifest_file data/qurating/smoke_split_manifest.json \
  --output_report reports/server/train_eval_overlap_report.json 2>&1 | tee -a "$OUTPUT_FILE"

# 7. Execute data direction and preference score polarity audits
echo "---------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Auditing preference label directions..." | tee -a "$OUTPUT_FILE"
python scripts/verify_data_direction.py 2>&1 | tee -a "$OUTPUT_FILE"

echo "---------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Data verification and split regeneration complete. Saved to $OUTPUT_FILE"
