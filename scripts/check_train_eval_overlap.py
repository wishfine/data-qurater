import json
import argparse
import sys
import os

def check_overlap(train_path, eval_path, output_report_path):
    print(f"=== CHECKING DATA OVERLAP BETWEEN TRAIN AND EVAL ===")
    print(f"Train File: {train_path}")
    print(f"Eval File:  {eval_path}")
    
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        print("[ERROR] Train or eval path does not exist.")
        sys.exit(1)
        
    train_pairs = []
    train_texts = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                a = item.get("text_a", "").strip()
                b = item.get("text_b", "").strip()
                if a and b:
                    train_pairs.append((a, b))
                    train_texts.add(a)
                    train_texts.add(b)
                    
    eval_pairs = []
    eval_texts = set()
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                a = item.get("text_a", "").strip()
                b = item.get("text_b", "").strip()
                if a and b:
                    eval_pairs.append((a, b))
                    eval_texts.add(a)
                    eval_texts.add(b)
                    
    # Exact pair duplicates
    train_pairs_set = set(train_pairs)
    exact_duplicates = []
    swapped_duplicates = []
    
    for ea, eb in eval_pairs:
        if (ea, eb) in train_pairs_set:
            exact_duplicates.append((ea, eb))
        if (eb, ea) in train_pairs_set:
            swapped_duplicates.append((ea, eb))
            
    # Text overlap
    shared_texts = train_texts.intersection(eval_texts)
    text_overlap_ratio = len(shared_texts) / max(len(eval_texts), 1)
    
    report = {
        "train_file": train_path,
        "eval_file": eval_path,
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "exact_duplicate_count": len(exact_duplicates),
        "swapped_duplicate_count": len(swapped_duplicates),
        "eval_unique_texts": len(eval_texts),
        "shared_text_count": len(shared_texts),
        "text_overlap_ratio": text_overlap_ratio
    }
    
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 50)
    print("OVERLAP CHECK SUMMARY")
    print("=" * 50)
    print(f"Train Pairs          : {len(train_pairs)}")
    print(f"Eval Pairs           : {len(eval_pairs)}")
    print(f"Exact Duplicates     : {len(exact_duplicates)}")
    print(f"Swapped Duplicates   : {len(swapped_duplicates)}")
    print(f"Shared Unique Texts  : {len(shared_texts)}")
    print(f"Eval Text Leak Ratio : {text_overlap_ratio:.4f}")
    print("=" * 50 + "\n")
    
    if len(exact_duplicates) > 0:
        raise ValueError(f"[CRITICAL ERROR] Data Leakage Detected! {len(exact_duplicates)} exact pairs overlap between train and eval sets.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_report", type=str, default="reports/server/train_eval_overlap_report.json")
    args = parser.parse_args()
    
    check_overlap(args.train_file, args.eval_file, args.output_report)
