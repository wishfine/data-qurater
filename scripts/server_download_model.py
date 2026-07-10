import os
import sys
import shutil

MODEL_ID = "Qwen/Qwen3.5-4B"
SHARED_TARGET_DIR = "/home/share_ssd_data/nfs-env/zhangyonglin7/model/qwen/Qwen3.5-4B"
LOCAL_CACHE_DIR = "/home/zhangyonglin/models"

os.makedirs("outputs", exist_ok=True)

# Helper to verify a model directory is valid
def is_valid_model_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

# 1. If it's already in the shared target directory, we use it directly!
if is_valid_model_dir(SHARED_TARGET_DIR):
    print(f"Found valid model in shared target storage: {SHARED_TARGET_DIR}")
    resolved_path = os.path.realpath(SHARED_TARGET_DIR)
    with open("outputs/model_path.txt", "w", encoding="utf-8") as f:
        f.write(resolved_path + "\n")
    print(f"Path written to outputs/model_path.txt")
    sys.exit(0)

# 2. If it's in the old local ModelScope cache, try to copy it to the shared target directory!
old_snapshot_path = os.path.join(LOCAL_CACHE_DIR, "models/Qwen--Qwen3.5-4B/snapshots/master")
if is_valid_model_dir(old_snapshot_path):
    print(f"Found model in local ModelScope cache: {old_snapshot_path}")
    print(f"Transferring model files to shared storage: {SHARED_TARGET_DIR} ...")
    try:
        os.makedirs(SHARED_TARGET_DIR, exist_ok=True)
        for item in os.listdir(old_snapshot_path):
            s = os.path.join(old_snapshot_path, item)
            d = os.path.join(SHARED_TARGET_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("Transfer complete.")
        resolved_path = os.path.realpath(SHARED_TARGET_DIR)
        with open("outputs/model_path.txt", "w", encoding="utf-8") as f:
            f.write(resolved_path + "\n")
        print(f"Path written to outputs/model_path.txt")
        sys.exit(0)
    except Exception as e:
        print(f"WARNING: Failed to transfer files to shared storage: {e}")
        # Fallback to local snapshot path
        resolved_path = os.path.realpath(old_snapshot_path)
        with open("outputs/model_path.txt", "w", encoding="utf-8") as f:
            f.write(resolved_path + "\n")
        print(f"Fallback path written to outputs/model_path.txt")
        sys.exit(0)

# 3. If it's not found in either, download it to the shared storage cache directory!
print(f"Model not found in target or local cache. Initiating download to shared storage...")
try:
    from modelscope import snapshot_download
    model_dir = snapshot_download(
        model_id=MODEL_ID,
        cache_dir="/home/share_ssd_data/nfs-env/zhangyonglin7/model/qwen",
    )
    resolved_path = os.path.realpath(model_dir)
    with open("outputs/model_path.txt", "w", encoding="utf-8") as f:
        f.write(resolved_path + "\n")
    print(f"Download complete. Path written to outputs/model_path.txt")
except Exception as e:
    print(f"[ERROR] Failed to download from ModelScope: {e}")
    sys.exit(1)
