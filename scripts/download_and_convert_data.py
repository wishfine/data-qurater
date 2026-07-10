import os
import json
import sys

# Ensure dataset module can be imported or instruct install
try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] 'datasets' library is not installed in the current python environment.")
    print("Please run: pip install datasets huggingface_hub")
    sys.exit(1)

from data.qurating_dataset import OfficialQuRatingDatasetAdapter

def dump_ds_to_jsonl(ds, path):
    # Retrieve the first available split name (e.g., 'train', 'test')
    split_name = list(ds.keys())[0]
    print(f"Found split: '{split_name}'. Dumping records to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for item in ds[split_name]:
            # Convert calibrated_predictions or any other fields to primitive types for json compliance
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    os.makedirs("data/qurating", exist_ok=True)

    # 1. Download official training judgments
    print("=== 1. DOWNLOADING PRINCETON QURATING TRAINING DATA ===")
    try:
        train_ds = load_dataset("princeton-nlp/QuRating-GPT3.5-Judgments")
    except Exception as e:
        print(f"[ERROR] Failed to download training dataset: {e}")
        sys.exit(1)

    temp_train_path = "data/qurating/temp_train.jsonl"
    dump_ds_to_jsonl(train_ds, temp_train_path)

    # 2. Download official evaluation judgments
    print("\n=== 2. DOWNLOADING PRINCETON QURATING TEST DATA ===")
    try:
        eval_ds = load_dataset("princeton-nlp/QuRating-GPT3.5-Judgments-Test")
    except Exception as e:
        print(f"[ERROR] Failed to download test dataset: {e}")
        sys.exit(1)

    temp_eval_path = "data/qurating/temp_eval.jsonl"
    dump_ds_to_jsonl(eval_ds, temp_eval_path)

    # 3. Convert them using the Normalized Adapter
    train_output = "data/qurating/train.jsonl"
    eval_output = "data/qurating/eval.jsonl"

    print("\n=== 3. CONVERTING TRAINING DATASET TO NORMALIZED PAIRWISE FORMAT ===")
    OfficialQuRatingDatasetAdapter.convert_file(temp_train_path, train_output)

    print("\n=== 4. CONVERTING EVALUATION DATASET TO NORMALIZED PAIRWISE FORMAT ===")
    OfficialQuRatingDatasetAdapter.convert_file(temp_eval_path, eval_output)

    # Clean up temporary raw files
    if os.path.exists(temp_train_path):
        os.remove(temp_train_path)
    if os.path.exists(temp_eval_path):
        os.remove(temp_eval_path)

    print("\n==================================================")
    print("[SUCCESS] Datasets downloaded and converted successfully!")
    print(f"  Standardized Training file   : {train_output}")
    print(f"  Standardized Evaluation file : {eval_output}")
    print("==================================================")

if __name__ == "__main__":
    main()
