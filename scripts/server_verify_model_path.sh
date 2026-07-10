#!/usr/bin/env bash
# server_verify_model_path.sh
# Verifies model path written in outputs/model_path.txt

set -euo pipefail

mkdir -p reports/server
OUTPUT_FILE="reports/server/model_path_verification.txt"

echo "=== LIGHTWEIGHT MODEL PATH VERIFICATION ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"

if [ ! -f "outputs/model_path.txt" ]; then
    echo "[ERROR] outputs/model_path.txt does not exist!" | tee -a "$OUTPUT_FILE"
    exit 1
fi

RAW_CONTENT="$(cat outputs/model_path.txt)"
echo "Raw file content : '$RAW_CONTENT'" | tee -a "$OUTPUT_FILE"

# Clean line endings
MODEL_PATH="$(tr -d '\r\n' < outputs/model_path.txt)"
echo "Resolved path    : '$MODEL_PATH'" | tee -a "$OUTPUT_FILE"

# Check path exists
if [ -d "$MODEL_PATH" ]; then
    echo "Path exists      : YES (Directory)" | tee -a "$OUTPUT_FILE"
else
    echo "Path exists      : NO" | tee -a "$OUTPUT_FILE"
    exit 1
fi

# Check config.json exists
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "config.json      : YES (Found)" | tee -a "$OUTPUT_FILE"
else
    echo "config.json      : NO (Missing)" | tee -a "$OUTPUT_FILE"
    exit 1
fi

# Check tokenizer files exist
TOKENIZER_FOUND="NO"
if [ -f "$MODEL_PATH/tokenizer.json" ] || [ -f "$MODEL_PATH/tokenizer_config.json" ]; then
    TOKENIZER_FOUND="YES (Found)"
fi
echo "Tokenizer files  : $TOKENIZER_FOUND" | tee -a "$OUTPUT_FILE"

if [ "$TOKENIZER_FOUND" = "NO" ]; then
    echo "[ERROR] Tokenizer files are missing." | tee -a "$OUTPUT_FILE"
    exit 1
fi

echo "[SUCCESS] Model path validation passed." | tee -a "$OUTPUT_FILE"
