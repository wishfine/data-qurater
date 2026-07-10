#!/bin/bash
# server_check_env.sh
# Checks server PyTorch, CUDA, and NLP library environment. Fails fast if constraints are not met.

set -euo pipefail

mkdir -p reports/server
OUTPUT_FILE="reports/server/environment_output.txt"

echo "=== SERVER ENVIRONMENT CHECK ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"

# Execute Python environment verifier block using the env python
python -c '
import sys
import json
import os
import importlib.metadata

report = {
    "python": sys.executable,
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "Unknown"),
    "expected_conda_env": "agent-rl",
    "conda_env_matches_expected": (os.environ.get("CONDA_DEFAULT_ENV", "") == "agent-rl"),
    "torch_version": "N/A",
    "torch_cuda": "N/A",
    "cuda_available": False,
    "gpu_count": 0,
    "gpu_name": "N/A",
    "transformers_version": "N/A",
    "peft_version": "N/A",
    "safetensors_version": "N/A",
    "modelscope_version": "N/A",
    "missing_packages": [],
    "remediation_commands": [],
    "flash_linear_attention_installed": False,
    "flash_linear_attention_version": None,
    "fla_core_installed": False,
    "fla_core_version": None,
    "fla_import_ok": False,
    "causal_conv1d_installed": False,
    "causal_conv1d_version": None,
    "causal_conv1d_import_ok": False,
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

# Check flash-linear-attention
try:
    report["flash_linear_attention_version"] = importlib.metadata.version("flash-linear-attention")
    report["flash_linear_attention_installed"] = True
except Exception:
    pass

# Check fla-core
try:
    report["fla_core_version"] = importlib.metadata.version("fla-core")
    report["fla_core_installed"] = True
except Exception:
    pass

# Check fla import
try:
    import fla
    report["fla_import_ok"] = True
except ImportError:
    pass

# Check causal-conv1d
try:
    report["causal_conv1d_version"] = importlib.metadata.version("causal-conv1d")
    report["causal_conv1d_installed"] = True
except Exception:
    pass

# Check causal-conv1d import
try:
    import causal_conv1d
    from causal_conv1d import causal_conv1d_fn
    report["causal_conv1d_import_ok"] = True
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
    report["missing_packages"].append("modelscope")
    report["remediation_commands"].append(
        "python -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
    )

# Since backbone is now Qwen3.5-4B, flash-linear-attention and causal-conv1d are fast-path dependencies
if not report["fla_import_ok"]:
    status = "FAIL"
    failures.append("flash-linear-attention missing (fast-path for Qwen3.5)")
    
if not report["causal_conv1d_import_ok"]:
    status = "FAIL"
    failures.append("causal-conv1d missing (fast-path for Qwen3.5)")

report["status"] = status

# Output status JSON
with open("reports/server/environment_status.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# Print warnings/logs
if not report["conda_env_matches_expected"]:
    print(f"WARNING: Conda environment name is \"{report[\"conda_env\"]}\", expected \"agent-rl\".")

if status == "FAIL":
    print(f"[CRITICAL ERROR] Hard environment checks failed: {failures}")
    if "modelscope" in report["missing_packages"]:
        print("\n=== REMEDIATION INSTRUCTIONS ===")
        print("ModelScope is missing. Please run the following command to install it:")
        print("  conda activate agent-rl")
        print("  python -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("\n*IMPORTANT*: Do NOT upgrade, downgrade, or reinstall PyTorch during this process!")
        print("================================")
    sys.exit(1)
else:
    print("[SUCCESS] Environment check passed. status=PASS")
' 2>&1 | tee -a "$OUTPUT_FILE"

echo "Environment check process completed." | tee -a "$OUTPUT_FILE"
