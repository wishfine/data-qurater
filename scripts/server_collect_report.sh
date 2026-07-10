#!/bin/bash
# server_collect_report.sh
# Compile server logs and verification reports into a unified summary.

set -u

OUTPUT_FILE="reports/server/server_verification_summary.txt"

echo "=== SERVER VERIFICATION SUMMARY REPORT (QWEN3.5-4B) ===" > "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "=======================================================" >> "$OUTPUT_FILE"

# 1. Model Download Check
if [ -f "reports/server/model_download_output.txt" ]; then
    echo "[MODEL DOWNLOAD] Log found." >> "$OUTPUT_FILE"
    grep -E "MODEL_ID|MODEL_PATH|Saved actual model path" reports/server/model_download_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[MODEL DOWNLOAD] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 2. Environment Check Report
if [ -f "reports/server/environment_output.txt" ]; then
    echo "[ENV CHECK] Log found." >> "$OUTPUT_FILE"
    grep -E "Python Path|Conda Environment|PyTorch Version|torch.cuda.is_available\(\)|GPU" reports/server/environment_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[ENV CHECK] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 3. Unit Test Report
if [ -f "reports/server/unit_test_output.txt" ]; then
    echo "[UNIT TESTS] Log found." >> "$OUTPUT_FILE"
    tail -n 10 reports/server/unit_test_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[UNIT TESTS] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 4. Data Direction Check Report
if [ -f "reports/server/data_direction_output.txt" ]; then
    echo "[DATA DIRECTION] Log found." >> "$OUTPUT_FILE"
    grep -E "VERIFICATION|Record #1:" -A 5 reports/server/data_direction_output.txt >> "$OUTPUT_FILE" || true
else
    echo "[DATA DIRECTION] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 5. Overlap Check Report
if [ -f "reports/server/train_eval_overlap_report.json" ]; then
    echo "[DATA OVERLAP CHECK] Report found." >> "$OUTPUT_FILE"
    cat reports/server/train_eval_overlap_report.json >> "$OUTPUT_FILE"
else
    echo "[DATA OVERLAP CHECK] Report NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 6. Baseline Evaluation Check
if [ -f "reports/server/baseline_eval_output.txt" ]; then
    echo "[BASELINE EVAL] Log found." >> "$OUTPUT_FILE"
    grep -E "Macro Accuracy|Accuracy          |Balanced Accuracy|BCE Loss|AUC" reports/server/baseline_eval_output.txt | head -n 15 >> "$OUTPUT_FILE" || true
else
    echo "[BASELINE EVAL] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 7. Smoke Test Report
if [ -f "reports/server/smoke_output.txt" ]; then
    echo "[SMOKE TEST] Log found." >> "$OUTPUT_FILE"
    grep -E "OPTIMIZER PARAMETER GROUPS|BENCHMARK STEP|Step Latency|Throughput|GPU Max Memory|GRAD VERIFY|micro_step" reports/server/smoke_output.txt | tail -n 25 >> "$OUTPUT_FILE" || true
else
    echo "[SMOKE TEST] Log NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 8. Checkpoint Round-Trip Report
if [ -f "reports/server/checkpoint_roundtrip.json" ]; then
    echo "[CHECKPOINT ROUND-TRIP] Report found." >> "$OUTPUT_FILE"
    cat reports/server/checkpoint_roundtrip.json >> "$OUTPUT_FILE"
else
    echo "[CHECKPOINT ROUND-TRIP] Report NOT found!" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# 9. Checkpoint Comparison Report
if [ -f "reports/server/training_comparison.md" ]; then
    echo "[TRAINING COMPARISON] Table found." >> "$OUTPUT_FILE"
    cat reports/server/training_comparison.md >> "$OUTPUT_FILE"
else
    echo "[TRAINING COMPARISON] Table NOT found!" >> "$OUTPUT_FILE"
fi
echo "=======================================================" >> "$OUTPUT_FILE"

echo "Summary report successfully generated at: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
