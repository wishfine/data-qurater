import os
import sys
from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen3-0.6B"
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

print("=" * 60)
print("MODEL_ID:", MODEL_ID)
print("MODEL_PATH:", model_dir)
print("=" * 60)

os.makedirs("outputs", exist_ok=True)
with open("outputs/model_path.txt", "w", encoding="utf-8") as f:
    f.write(model_dir.strip())
print(f"Saved actual model path to outputs/model_path.txt: {model_dir.strip()}")
