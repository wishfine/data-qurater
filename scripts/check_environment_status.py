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
        
    if data.get("status") != "PASS":
        print("[CRITICAL ERROR] Environment verification has not passed.")
        print(f"Conda Env Name: {data.get('conda_env', 'Unknown')}")
        print("Please check details in reports/server/environment_output.txt")
        sys.exit(1)
        
    print("[SUCCESS] Environment check verification passed.")

if __name__ == "__main__":
    main()
