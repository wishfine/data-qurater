#!/bin/bash
# server_verify_data.sh
# Run pairwise data format and direction verification.

set -euo pipefail

mkdir -p reports/server
OUTPUT_FILE="reports/server/data_direction_output.txt"

echo "=== RUNNING DATA FORMAT AND DIRECTION VERIFICATION ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "-----------------------------------------------------" | tee -a "$OUTPUT_FILE"

# Execute data direction checks
python3 scripts/verify_data_direction.py 2>&1 | tee -a "$OUTPUT_FILE"

echo "-----------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Data verification complete. Saved to $OUTPUT_FILE"
