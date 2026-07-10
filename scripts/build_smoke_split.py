import os
import sys
import json
import random
import hashlib
from collections import defaultdict

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

def normalize_text(text):
    return " ".join(text.strip().split())

def build_split(source_file, train_file, eval_file, manifest_file, raw_train_target=4, raw_eval_target=4, seed=42):
    print("=== PARTITIONING SMOKE TRAIN AND EVAL DATA (TEXT-DISJOINT) ===")
    print(f"Source file   : {source_file}")
    print(f"Train output  : {train_file}")
    print(f"Eval output   : {eval_file}")
    print(f"Manifest output: {manifest_file}")
    
    random.seed(seed)
    
    if not os.path.exists(source_file):
        print(f"[ERROR] Source file not found: {source_file}")
        sys.exit(1)
        
    # Read raw pairwise items
    raw_items = []
    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_items.append(json.loads(line))
                
    # Normalize texts for each raw item
    for item in raw_items:
        item["norm_a"] = normalize_text(item["text_a"])
        item["norm_b"] = normalize_text(item["text_b"])
        
    # Build Graph: find connected components (无向图连通分量)
    adj = defaultdict(list)
    for idx, item in enumerate(raw_items):
        u = item["norm_a"]
        v = item["norm_b"]
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    visited_nodes = set()
    components = []  # list of lists of raw item indices
    
    for item in raw_items:
        start_node = item["norm_a"]
        if start_node in visited_nodes:
            continue
            
        # BFS to find connected component (连通分量)
        comp_item_indices = set()
        queue = [start_node]
        visited_nodes.add(start_node)
        
        while queue:
            node = queue.pop(0)
            for neighbor, item_idx in adj[node]:
                comp_item_indices.add(item_idx)
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
                    
        components.append(list(comp_item_indices))
        
    print(f"Detected {len(components)} text-connected components (连通分量).")
    
    # Shuffle components
    random.shuffle(components)
    
    train_raw_indices = []
    eval_raw_indices = []
    unused_components_count = 0
    
    # Partition components
    for comp in components:
        if len(train_raw_indices) < raw_train_target:
            train_raw_indices.extend(comp)
        elif len(eval_raw_indices) < raw_eval_target:
            eval_raw_indices.extend(comp)
        else:
            unused_components_count += 1
            
    # Verify count targets
    if len(train_raw_indices) < raw_train_target or len(eval_raw_indices) < raw_eval_target:
        print(f"WARNING: Could not achieve target split of {raw_train_target}/{raw_eval_target} raw pairs due to connected component constraints.")
        print(f"Actual Train raw pairs: {len(train_raw_indices)} | Actual Eval raw pairs: {len(eval_raw_indices)}")
        if len(train_raw_indices) == 0 or len(eval_raw_indices) == 0:
            raise ValueError("[CRITICAL ERROR] Partition results in empty train or eval set!")
            
    # Expand to normalized records
    train_records = []
    train_texts = set()
    for idx in train_raw_indices:
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
            train_texts.add(item["norm_a"])
            train_texts.add(item["norm_b"])
            
    eval_records = []
    eval_texts = set()
    for idx in eval_raw_indices:
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
            eval_texts.add(item["norm_a"])
            eval_texts.add(item["norm_b"])
            
    # Hard assertion of disjoint sets
    shared_texts = train_texts.intersection(eval_texts)
    assert len(shared_texts) == 0, f"[CRITICAL ERROR] Text leak detected! Shared texts: {shared_texts}"
    
    # Save files
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(eval_file, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Save manifest
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
    manifest = {
        "seed": seed,
        "source_file": source_file,
        "source_sha256": get_sha256(source_file),
        "train_file": train_file,
        "train_sha256": get_sha256(train_file),
        "eval_file": eval_file,
        "eval_sha256": get_sha256(eval_file),
        "raw_train_pair_count": len(train_raw_indices),
        "raw_eval_pair_count": len(eval_raw_indices),
        "train_size": len(train_records),
        "eval_size": len(eval_records),
        "train_unique_text_count": len(train_texts),
        "eval_unique_text_count": len(eval_texts),
        "shared_text_count": len(shared_texts),
        "single_text_overlap_ratio": len(shared_texts) / max(len(eval_texts), 1),
        "component_count": len(components),
        "unused_component_count": unused_components_count,
        "exact_pair_overlap": 0,
        "swapped_pair_overlap": 0
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"[SUCCESS] Disjoint split complete. Train records: {len(train_records)}, Eval records: {len(eval_records)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="data/qurating/smoke_train_source.jsonl")
    parser.add_argument("--train_file", type=str, default="data/qurating/smoke_train.jsonl")
    parser.add_argument("--eval_file", type=str, default="data/qurating/smoke_eval.jsonl")
    parser.add_argument("--manifest_file", type=str, default="data/qurating/smoke_split_manifest.json")
    parser.add_argument("--train_size", type=int, default=4, help="Target number of raw train connected components")
    parser.add_argument("--eval_size", type=int, default=4, help="Target number of raw eval connected components")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    build_split(
        args.source_file,
        args.train_file,
        args.eval_file,
        args.manifest_file,
        args.train_size,
        args.eval_size,
        args.seed
    )
