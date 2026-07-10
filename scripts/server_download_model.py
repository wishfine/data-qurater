import os
import sys
from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen3.5-4B"
CACHE_DIR = "/home/zhangyonglin/models"

os.makedirs(CACHE_DIR, exist_ok=True)

try:
    model_dir = snapshot_download(
        model_id=MODEL_ID,
        cache_dir=CACHE_DIR,
    )
except Exception as e:
    print(f"[ERROR] Failed to download {MODEL_ID} from ModelScope: {e}")
    sys.exit(1)

# Resolve real absolute path
resolved_path = os.path.realpath(model_dir)

# Assertions to prevent invalid model paths from writing
assert os.path.isabs(resolved_path), f"Path is not absolute: {resolved_path}"
assert os.path.isdir(resolved_path), f"Path is not a directory: {resolved_path}"
assert os.path.isfile(os.path.join(resolved_path, "config.json")), f"config.json missing in {resolved_path}"

# Write ONLY the absolute path, with no decorators/logs/headers
os.makedirs("outputs", exist_ok=True)
with open("outputs/model_path.txt", "w", encoding="utf-8") as f:
    f.write(resolved_path + "\n")

print(f"Successfully downloaded and verified. Path written to outputs/model_path.txt")
