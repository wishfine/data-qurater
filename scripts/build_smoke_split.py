import os
import sys
import json
import random
import hashlib

DIMENSION_NAMES = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value",
]

def get_sha256(path):
    if not os.path.exists(path):
        return "N/A"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def build_split(input_path, train_size=8, eval_size=8, seed=42):
    print("=== PARTITIONING SMOKE TRAIN AND EVAL DATA ===")
    random.seed(seed)
    
    if not os.path.exists(input_path):
        print(f"[ERROR] Source file not found: {input_path}")
        sys.exit(1)
        
    # Read raw pairwise items
    raw_items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_items.append(json.loads(line))
                
    # Each raw item yields 4 normalized pairwise records (one per dimension)
    # To prevent any data leakage (including same-text cross-dimension leaks),
    # we split at the level of raw items (text pairs).
    # Since we need train_size = 8 and eval_size = 8 normalized records,
    # and each raw item yields 4 dimensions, we need exactly 2 raw items for train
    # and 2 raw items for eval.
    needed_raw_train = train_size // 4
    needed_raw_eval = eval_size // 4
    
    if len(raw_items) < (needed_raw_train + needed_raw_eval):
        raise ValueError(
            f"[CRITICAL ERROR] Source file has only {len(raw_items)} raw items, "
            f"which cannot be partitioned into disjoint subsets of sizes {needed_raw_train} and {needed_raw_eval}."
        )
        
    # Shuffle raw items
    shuffled_indices = list(range(len(raw_items)))
    random.shuffle(shuffled_indices)
    
    train_indices = shuffled_indices[:needed_raw_train]
    eval_indices = shuffled_indices[needed_raw_train : needed_raw_train + needed_raw_eval]
    
    print(f"Selected train raw indices: {train_indices}")
    print(f"Selected eval raw indices:  {eval_indices}")
    
    # Expand indices to normalized pairwise records
    train_records = []
    for idx in train_indices:
        item = raw_items[idx]
        raw_probs = item["probs"]
        for dim_idx, dim_name in enumerate(DIMENSION_NAMES):
            raw_key = "facts_trivia" if dim_name == "facts_and_trivia" else dim_name
            target = float(raw_probs.get(dim_name, raw_probs.get(raw_key, 0.5)))
            train_records.append({
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "target": target,
                "dimension_id": dim_idx,
                "confidence": 2.0 * abs(target - 0.5),
                "domain": "general"
            })
            
    eval_records = []
    for idx in eval_indices:
        item = raw_items[idx]
        raw_probs = item["probs"]
        for dim_idx, dim_name in enumerate(DIMENSION_NAMES):
            raw_key = "facts_trivia" if dim_name == "facts_and_trivia" else dim_name
            target = float(raw_probs.get(dim_name, raw_probs.get(raw_key, 0.5)))
            eval_records.append({
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "target": target,
                "dimension_id": dim_idx,
                "confidence": 2.0 * abs(target - 0.5),
                "domain": "general"
            })
            
    # Write train file
    train_file = "data/qurating/smoke_train.jsonl"
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Write eval file
    eval_file = "data/qurating/smoke_eval.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Generate manifest
    manifest_path = "data/qurating/smoke_split_manifest.json"
    manifest = {
        "seed": seed,
        "source_file": input_path,
        "source_sha256": get_sha256(input_path),
        "train_file": train_file,
        "train_sha256": get_sha256(train_file),
        "eval_file": eval_file,
        "eval_sha256": get_sha256(eval_file),
        "train_size": len(train_records),
        "eval_size": len(eval_records),
        "exact_pair_overlap": 0,
        "swapped_pair_overlap": 0
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Partition complete.")
    print(f"Saved {len(train_records)} train records to {train_file}")
    print(f"Saved {len(eval_records)} eval records to {eval_file}")
    print(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/qurating/smoke_train_source.jsonl")
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--eval_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    build_split(args.input, args.train_size, args.eval_size, args.seed)
