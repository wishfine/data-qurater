#!/bin/bash
# server_collect_report.sh
# Compile server logs and verification reports into a unified summary.

set -u

OUTPUT_FILE="reports/server/server_verification_summary.txt"

echo "=== SERVER VERIFICATION SUMMARY REPORT ===" > "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"

# 1. Environment Check Report
if [ -f "reports/server/environment_output.txt" ]; then
    echo "[ENV CHECK] Log found." >> "$OUTPUT_FILE"
    grep -E "Python Path|Conda Environment|PyTorch Version|torch.cuda.is_available\(\)|GPU" reports/server/environment_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[ENV CHECK] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 2. Unit Test Report
if [ -f "reports/server/unit_test_output.txt" ]; then
    echo "[UNIT TESTS] Log found." >> "$OUTPUT_FILE"
    tail -n 10 reports/server/unit_test_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[UNIT TESTS] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 3. Data Direction Check Report
if [ -f "reports/server/data_direction_output.txt" ]; then
    echo "[DATA DIRECTION] Log found." >> "$OUTPUT_FILE"
    grep -E "VERIFICATION|Record #1:" -A 5 reports/server/data_direction_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[DATA DIRECTION] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 4. Smoke Test Report
if [ -f "reports/server/smoke_output.txt" ]; then
    echo "[SMOKE TEST] Log found." >> "$OUTPUT_FILE"
    grep -E "BENCHMARK STEP|Step Latency|Throughput|GPU Max Memory" reports/server/smoke_output.txt | tail -n 15 >> "$OUTPUT_FILE" || true
else
    echo "[SMOKE TEST] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 5. Checkpoint Load Report
if [ -f "reports/server/checkpoint_output.txt" ]; then
    echo "[CHECKPOINT LOAD] Log found." >> "$OUTPUT_FILE"
    tail -n 15 reports/server/checkpoint_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[CHECKPOINT LOAD] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "==========================================" >> "$OUTPUT_FILE"

echo "Summary report successfully generated at: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
