#!/bin/bash
# server_run_unit_tests.sh
# Run unit tests on the server and preserve exit code.

set -euo pipefail

mkdir -p reports/server
OUTPUT_FILE="reports/server/unit_test_output.txt"

echo "=== RUNNING SERVER-SIDE UNIT TESTS ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "--------------------------------------" | tee -a "$OUTPUT_FILE"

# Run tests and pipe output, preserving test exit code via pipefail
python3 -m unittest discover -s tests -p "test_*.py" -v 2>&1 | tee -a "$OUTPUT_FILE"

echo "--------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Unit tests execution complete. Saved to $OUTPUT_FILE"
