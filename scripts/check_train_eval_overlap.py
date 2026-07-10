import os
import sys
import json
import hashlib

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

def check_overlap(train_path, eval_path, manifest_path, output_report_path):
    print("=== RUNNING ENHANCED TRAIN-EVAL OVERLAP & LEAKAGE CHECK ===")
    
    train_sha = get_sha256(train_path)
    eval_sha = get_sha256(eval_path)
    same_file_content = (train_sha == eval_sha and train_sha != "N/A")
    
    # 1. Parse manifest and verify hashes
    manifest_matches_current_files = True
    source_hash_matches = False
    train_hash_matches = False
    eval_hash_matches = False
    
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            source_file = manifest.get("source_file", "")
            source_sha = get_sha256(source_file)
            
            source_hash_matches = (manifest.get("source_sha256") == source_sha)
            train_hash_matches = (manifest.get("train_sha256") == train_sha)
            eval_hash_matches = (manifest.get("eval_sha256") == eval_sha)
            
            if not (source_hash_matches and train_hash_matches and eval_hash_matches):
                manifest_matches_current_files = False
        except Exception as e:
            print(f"[ERROR] Failed to load or verify manifest {manifest_path}: {e}")
            manifest_matches_current_files = False
    else:
        print(f"[WARNING] Manifest file not found: {manifest_path}")
        manifest_matches_current_files = False
        
    train_records = []
    train_texts = set()
    train_pairs = set()  # (text_a, text_b)
    
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    train_records.append(item)
                    a = normalize_text(item.get("text_a", ""))
                    b = normalize_text(item.get("text_b", ""))
                    if a and b:
                        train_pairs.add((a, b))
                        train_texts.add(a)
                        train_texts.add(b)
                        
    eval_records = []
    eval_texts = set()
    eval_pairs = set()
    
    exact_pair_overlap_count = 0
    swapped_pair_overlap_count = 0
    
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_records.append(item)
                    a = normalize_text(item.get("text_a", ""))
                    b = normalize_text(item.get("text_b", ""))
                    
                    if a and b:
                        eval_pairs.add((a, b))
                        eval_texts.add(a)
                        eval_texts.add(b)
                        
                        # Check 1: Exact pair overlap
                        if (a, b) in train_pairs:
                            exact_pair_overlap_count += 1
                        # Check 2: Swapped pair overlap
                        if (b, a) in train_pairs:
                            swapped_pair_overlap_count += 1

    shared_texts = train_texts.intersection(eval_texts)
    shared_text_count = len(shared_texts)
    single_text_overlap_ratio = shared_text_count / max(len(eval_texts), 1)
    
    # Hard failure conditions:
    # 1. Manifest verification fails
    # 2. Same file content
    # 3. Exact pair overlap
    # 4. Swapped pair overlap
    # 5. Shared single text count > 0 (Leakage)
    has_failed = (
        not manifest_matches_current_files
        or same_file_content
        or exact_pair_overlap_count > 0
        or swapped_pair_overlap_count > 0
        or shared_text_count > 0
    )
    
    status = "FAIL" if has_failed else "PASS"
    
    report = {
        "train_sha256": train_sha,
        "eval_sha256": eval_sha,
        "same_file_content": same_file_content,
        "exact_pair_overlap_count": exact_pair_overlap_count,
        "swapped_pair_overlap_count": swapped_pair_overlap_count,
        "shared_text_count": shared_text_count,
        "single_text_overlap_ratio": single_text_overlap_ratio,
        "manifest_matches_current_files": manifest_matches_current_files,
        "source_hash_matches": source_hash_matches,
        "train_hash_matches": train_hash_matches,
        "eval_hash_matches": eval_hash_matches,
        "status": status
    }
    
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 50)
    print("ENHANCED DATA LEAKAGE AUDIT REPORT")
    print("=" * 50)
    print(f"Train File SHA256     : {train_sha}")
    print(f"Eval File SHA256      : {eval_sha}")
    print(f"Same File Content     : {same_file_content}")
    print(f"Exact Pair Overlaps   : {exact_pair_overlap_count}")
    print(f"Swapped Pair Overlaps : {swapped_pair_overlap_count}")
    print(f"Shared Single Texts   : {shared_text_count}")
    print(f"Text Overlap Ratio    : {single_text_overlap_ratio:.4f}")
    print(f"Manifest Match Status : {manifest_matches_current_files}")
    print(f"Audit Status          : {status}")
    print("=" * 50 + "\n")
    
    if has_failed:
        print("[CRITICAL ERROR] Train/Eval data leakage check failed! Hard failures triggered.")
        sys.exit(1)
    else:
        print("[SUCCESS] Data split has passed all overlap and leakage checks.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--manifest_file", type=str, default="data/qurating/smoke_split_manifest.json")
    parser.add_argument("--output_report", type=str, default="reports/server/train_eval_overlap_report.json")
    args = parser.parse_args()
    
    check_overlap(args.train_file, args.eval_file, args.manifest_file, args.output_report)
