#!/usr/bin/env bash
# server_download_model.sh
# Run model downloader script and verify generated path.

set -euo pipefail

# Environment Pre-check via python helper
python3 scripts/check_environment_status.py

mkdir -p reports/server outputs

echo "=== STARTING MODELSCOPE DOWNLOAD ==="
python3 scripts/server_download_model.py 2>&1 | tee reports/server/model_download_output.txt

# Post-download path verification
if [ -f "outputs/model_path.txt" ]; then
    MODEL_PATH="$(tr -d '\r\n' < outputs/model_path.txt)"
    
    echo "Verifying downloaded path: '$MODEL_PATH'"
    
    # Assertions
    test -n "$MODEL_PATH"
    test -d "$MODEL_PATH"
    test -f "$MODEL_PATH/config.json"
    
    echo "[SUCCESS] Model download and verification passed."
else
    echo "[ERROR] outputs/model_path.txt was not generated!"
    exit 1
fi
echo "=== DOWNLOAD PROCESS FINISHED ==="
