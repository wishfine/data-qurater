#!/bin/bash
# server_check_env.sh
# Checks server PyTorch, CUDA, and NLP library environment. Fails fast if constraints are not met.

set -euo pipefail

mkdir -p reports/server
OUTPUT_FILE="reports/server/environment_output.txt"

echo "=== SERVER ENVIRONMENT CHECK ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"

# Execute Python environment verifier block
python3 -c '
import sys
import json
import os

report = {
    "python": sys.executable,
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "Unknown"),
    "torch_version": "N/A",
    "torch_cuda": "N/A",
    "cuda_available": False,
    "gpu_count": 0,
    "gpu_name": "N/A",
    "transformers_version": "N/A",
    "peft_version": "N/A",
    "safetensors_version": "N/A",
    "modelscope_version": "N/A",
    "status": "FAIL"
}

# Check torch
try:
    import torch
    report["torch_version"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu_count"] = torch.cuda.device_count()
        report["gpu_name"] = torch.cuda.get_device_name(0)
except ImportError:
    pass

# Check transformers
try:
    import transformers
    report["transformers_version"] = transformers.__version__
except ImportError:
    pass

# Check peft
try:
    import peft
    report["peft_version"] = peft.__version__
except ImportError:
    pass

# Check safetensors
try:
    import safetensors
    report["safetensors_version"] = safetensors.__version__
except ImportError:
    pass

# Check modelscope
try:
    import modelscope
    report["modelscope_version"] = modelscope.__version__
except ImportError:
    pass

# Assert verification conditions
status = "PASS"
failures = []

if report["torch_version"] == "N/A":
    status = "FAIL"
    failures.append("torch missing")
if not report["cuda_available"]:
    status = "FAIL"
    failures.append("CUDA unavailable")
if report["gpu_count"] == 0:
    status = "FAIL"
    failures.append("GPU count is 0")
if report["transformers_version"] == "N/A":
    status = "FAIL"
    failures.append("transformers missing")
if report["peft_version"] == "N/A":
    status = "FAIL"
    failures.append("peft missing")
if report["safetensors_version"] == "N/A":
    status = "FAIL"
    failures.append("safetensors missing")
if report["modelscope_version"] == "N/A":
    status = "FAIL"
    failures.append("modelscope missing")

report["status"] = status

# Output status JSON
with open("reports/server/environment_status.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# Print warnings/logs
if report["conda_env"] != "agent-rl":
    print(f"WARNING: Conda environment name is \"{report[\"conda_env\"]}\", expected \"agent-rl\".")

if status == "FAIL":
    print(f"[CRITICAL ERROR] Hard environment checks failed: {failures}")
    sys.exit(1)
else:
    print("[SUCCESS] Environment check passed. status=PASS")
' 2>&1 | tee -a "$OUTPUT_FILE"

echo "Environment check process completed." | tee -a "$OUTPUT_FILE"
