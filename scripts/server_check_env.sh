#!/bin/bash
# server_check_env.sh
# Check and report GPU, PyTorch, and linear attention environment status.

set -euo pipefail

mkdir -p reports/server
OUTPUT_FILE="reports/server/environment_output.txt"

echo "=== ENVIRONMENT VERIFICATION FOR QWENQURATER ===" | tee "$OUTPUT_FILE"
echo "Timestamp: $(date)" | tee -a "$OUTPUT_FILE"
echo "------------------------------------------------" | tee -a "$OUTPUT_FILE"

# 1. Paths and Conda Environment
echo "Python Path: $(which python3)" | tee -a "$OUTPUT_FILE"
echo "Conda Environment: ${CONDA_DEFAULT_ENV:-None}" | tee -a "$OUTPUT_FILE"

# 2. PyTorch and CUDA check
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'torch.version.cuda: {torch.version.cuda}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Devices Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('GPU Devices Count: 0')
" 2>&1 | tee -a "$OUTPUT_FILE"

# 3. Core NLP & Quantization Libraries check
python3 -c "
libs = ['transformers', 'peft', 'bitsandbytes', 'triton']
for lib in libs:
    try:
        mod = __import__(lib)
        print(f'{lib:<15} Version: {getattr(mod, \"__version__\", \"Installed (No version attr)\")}')
    except ImportError:
        print(f'{lib:<15} Version: NOT INSTALLED')
" 2>&1 | tee -a "$OUTPUT_FILE"

# 4. FLA Linear Attention library check
python3 -c "
try:
    import fla
    print('fla library     : SUCCESSFULLY IMPORTED (Linear Attention fast path is AVAILABLE)')
except ImportError:
    print('fla library     : NOT FOUND (Linear Attention fast path is UNAVAILABLE, will fallback)')
" 2>&1 | tee -a "$OUTPUT_FILE"

# 5. Pip check
echo "------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Running pip check..." | tee -a "$OUTPUT_FILE"
pip check 2>&1 | tee -a "$OUTPUT_FILE" || echo "Pip check returned warning/error (non-blocking)." | tee -a "$OUTPUT_FILE"

echo "------------------------------------------------" | tee -a "$OUTPUT_FILE"
echo "Environment verification complete. Saved to $OUTPUT_FILE"
