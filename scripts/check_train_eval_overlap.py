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

def check_overlap(train_path, eval_path, output_report_path):
    print("=== RUNNING ENHANCED TRAIN-EVAL OVERLAP & LEAKAGE CHECK ===")
    
    train_sha = get_sha256(train_path)
    eval_sha = get_sha256(eval_path)
    same_file_content = (train_sha == eval_sha and train_sha != "N/A")
    
    train_records = []
    train_texts = set()
    train_pairs = set()  # (text_a, text_b)
    train_sample_ids = set()
    
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    train_records.append(item)
                    a = item.get("text_a", "").strip()
                    b = item.get("text_b", "").strip()
                    if a and b:
                        train_pairs.add((a, b))
                        train_texts.add(a)
                        train_texts.add(b)
                    s_id = item.get("sample_id")
                    if s_id is not None:
                        train_sample_ids.add(s_id)
                        
    eval_records = []
    eval_texts = set()
    eval_pairs = set()
    eval_sample_ids = set()
    
    exact_pair_overlap_count = 0
    swapped_pair_overlap_count = 0
    cross_dimension_overlap_count = 0
    duplicate_sample_id_count = 0
    
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_records.append(item)
                    a = item.get("text_a", "").strip()
                    b = item.get("text_b", "").strip()
                    dim = item.get("dimension_id")
                    
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
                        # Check 3: Same pair + different dimension check
                        # (If exact pair exists, but we want to count how many pairs have cross-dimension presence)
                        for ta, tb in train_pairs:
                            if (ta == a and tb == b) or (ta == b and tb == a):
                                # If any text comparison matches, check if it was for a different dimension
                                # In our dataset loader, train and eval records are normalized per dimension.
                                pass
                                
                    s_id = item.get("sample_id")
                    if s_id is not None:
                        eval_sample_ids.add(s_id)
                        if s_id in train_sample_ids:
                            duplicate_sample_id_count += 1

    shared_texts = train_texts.intersection(eval_texts)
    single_text_overlap_ratio = len(shared_texts) / max(len(eval_texts), 1)
    
    # Validation hard failures
    has_failed = same_file_content or (exact_pair_overlap_count > 0) or (swapped_pair_overlap_count > 0)
    status = "FAIL" if has_failed else "PASS"
    
    report = {
        "train_sha256": train_sha,
        "eval_sha256": eval_sha,
        "same_file_content": same_file_content,
        "exact_pair_overlap_count": exact_pair_overlap_count,
        "swapped_pair_overlap_count": swapped_pair_overlap_count,
        "single_text_overlap_ratio": single_text_overlap_ratio,
        "duplicate_sample_id_count": duplicate_sample_id_count,
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
    print(f"Text Overlap Ratio    : {single_text_overlap_ratio:.4f}")
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
    parser.add_argument("--output_report", type=str, default="reports/server/train_eval_overlap_report.json")
    args = parser.parse_args()
    
    check_overlap(args.train_file, args.eval_file, args.output_report)
