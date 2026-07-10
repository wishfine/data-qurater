#!/usr/bin/env bash
# server_download_model.sh
# Run model downloader script and tee logs.

set -euo pipefail

mkdir -p reports/server outputs

echo "=== STARTING MODELSCOPE DOWNLOAD ==="
python3 scripts/server_download_model.py 2>&1 | tee reports/server/model_download_output.txt
echo "=== DOWNLOAD PROCESS FINISHED ==="
