import json
import sys
import os

def main():
    status_path = "reports/server/environment_status.json"
    if not os.path.exists(status_path):
        print(f"[ERROR] Environment status file does not exist: {status_path}")
        print("Please execute verification first: bash scripts/server_check_env.sh")
        sys.exit(1)
        
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read or parse environment status JSON: {e}")
        sys.exit(1)
        
    expected_env = data.get("expected_conda_env", "agentgym")
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    if current_env != expected_env:
        print(f"[CRITICAL ERROR] Current Conda environment is {current_env!r}, expected {expected_env!r}")
        sys.exit(1)

    if data.get("status") != "PASS":
        print("[CRITICAL ERROR] Stored environment verification status is not PASS")
        sys.exit(1)

    # Check python executable path alignment
    current_py = os.path.realpath(sys.executable)
    stored_py = data.get("python", "")
    if stored_py:
        stored_py_real = os.path.realpath(stored_py)
        if current_py != stored_py_real:
            print(f"[CRITICAL ERROR] Current python executable path ({current_py}) does not match stored verification path ({stored_py_real})")
            sys.exit(1)

    # Live check on PyTorch and CUDA
    try:
        import torch
    except ImportError:
        print("[CRITICAL ERROR] torch is not installable in the current Python process")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[CRITICAL ERROR] CUDA is not available in the current Python process")
        sys.exit(1)

    if torch.cuda.device_count() <= 0:
        print("[CRITICAL ERROR] No CUDA devices are visible")
        sys.exit(1)

    print("Stored environment status: PASS")
    print(f"Current environment: {current_env}")
    print(f"Current Python: {current_py}")
    print("Current CUDA available: True")
    print("Current environment verification: PASS")

if __name__ == "__main__":
    main()
