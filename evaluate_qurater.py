from __future__ import annotations
import os
import sys
import json
import argparse
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Any

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from data.qurating_dataset import NormalizedPairwiseDataset

def get_sha256(path):
    if not os.path.exists(path):
        return "N/A"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    pos_labels = (y_true > 0.5).astype(int)
    unique_classes = np.unique(pos_labels)
    
    pos_count = int(np.sum(pos_labels))
    neg_count = len(pos_labels) - pos_count
    
    if len(unique_classes) < 2:
        return {
            "auc": None,
            "auc_status": "UNDEFINED_SINGLE_CLASS",
            "positive_count": pos_count,
            "negative_count": neg_count
        }
        
    sorted_indices = np.argsort(y_scores)
    pos_labels_sorted = pos_labels[sorted_indices]
    
    ranks = np.arange(1, len(pos_labels) + 1)
    pos_ranks_sum = np.sum(ranks * pos_labels_sorted)
    
    auc = (pos_ranks_sum - (pos_count * (pos_count + 1)) / 2) / (pos_count * neg_count)
    return {
        "auc": float(auc),
        "auc_status": "OK",
        "positive_count": pos_count,
        "negative_count": neg_count
    }

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    pos_sum = np.sum(pos_mask)
    neg_sum = np.sum(neg_mask)
    
    if pos_sum == 0 or neg_sum == 0:
        return None
        
    tp_rate = np.mean(y_pred[pos_mask] == 1)
    tn_rate = np.mean(y_pred[neg_mask] == 0)
    
    return float(0.5 * (tp_rate + tn_rate))

def compute_confidence_buckets(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float | None | str]:
    confidences = 2.0 * np.abs(targets - 0.5)
    
    low_mask = (confidences < 0.3)
    med_mask = (confidences >= 0.3) & (confidences < 0.7)
    high_mask = (confidences >= 0.7)
    
    def get_acc(mask):
        if np.sum(mask) == 0:
            return None
        pred_label = (preds[mask] > 0.5).astype(int)
        gt_label = (targets[mask] > 0.5).astype(int)
        return float(np.mean(pred_label == gt_label))
        
    return {
        "low": get_acc(low_mask),
        "medium": get_acc(med_mask),
        "high": get_acc(high_mask)
    }

def get_distribution_stats(arr: np.ndarray) -> Dict[str, float | None]:
    if len(arr) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }

def clean_nan_inf(val: Any) -> Any:
    if isinstance(val, dict):
        return {k: clean_nan_inf(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [clean_nan_inf(v) for v in val]
    elif isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    return val

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate QwenQuRater quality predictor")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model directory")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to modular checkpoint directory")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--output_file", type=str, default=None, help="Save evaluation metrics summary to this json file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Verify checkpoint directory configuration exists
    config_path = os.path.join(args.checkpoint_dir, "qurater_config.json")
    if not os.path.exists(config_path):
        print(f"[CRITICAL ERROR] qurater_config.json is missing in: {args.checkpoint_dir}")
        sys.exit(1)

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer_dir = os.path.join(args.checkpoint_dir, "tokenizer")
    if os.path.exists(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    model = QwenQuRater(backbone=backbone)
    
    # Load LoRA adapter if present
    adapter_dir = os.path.join(args.checkpoint_dir, "adapter")
    with open(config_path, "r") as f:
        q_config = json.load(f)
    use_lora = q_config.get("use_lora", False)
    if use_lora:
        if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {adapter_dir} ...")
            model.backbone = PeftModel.from_pretrained(model.backbone, adapter_dir)
        else:
            print(f"[CRITICAL ERROR] Adapter config file is missing in {adapter_dir}")
            sys.exit(1)
            
    # Load rating head
    heads_path = os.path.join(args.checkpoint_dir, "rating_head.safetensors")
    if os.path.exists(heads_path):
        from safetensors.torch import load_file
        model.score.load_state_dict(load_file(heads_path, map_location=device))
    else:
        pt_path = os.path.join(args.checkpoint_dir, "rating_head.pt")
        if os.path.exists(pt_path):
            model.score.load_state_dict(torch.load(pt_path, map_location=device))
        else:
            print(f"[CRITICAL ERROR] Rating head weights missing in {args.checkpoint_dir}")
            sys.exit(1)
        
    model.to(device)
    model.eval()

    # 2. Load Evaluation Dataset
    print(f"Loading dataset from: {args.eval_file}")
    dataset = NormalizedPairwiseDataset(args.eval_file, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # 3. Running evaluation predictions
    all_targets = {dim_idx: [] for dim_idx in range(4)}
    all_preds = {dim_idx: [] for dim_idx in range(4)}
    all_diffs = {dim_idx: [] for dim_idx in range(4)}
    all_confidences = {dim_idx: [] for dim_idx in range(4)}
    domain_data = []

    print("Evaluating samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            targets = batch["targets"].to(device)
            dimension_ids = batch["dimension_ids"].to(device)
            confidences = batch["confidences"].to(device)
            domains = batch["domains"]
            
            ratings_a = model(input_ids_a, attention_mask_a)
            ratings_b = model(input_ids_b, attention_mask_b)
            
            for i in range(input_ids_a.size(0)):
                dim_idx = int(dimension_ids[i].item())
                r_a = ratings_a[i, dim_idx].cpu().float().item()
                r_b = ratings_b[i, dim_idx].cpu().float().item()
                gt = targets[i].cpu().float().item()
                
                logit = r_b - r_a
                p_pred = 1.0 / (1.0 + np.exp(-logit))
                
                all_targets[dim_idx].append(gt)
                all_preds[dim_idx].append(p_pred)
                all_diffs[dim_idx].append(logit)
                all_confidences[dim_idx].append(confidences[i].cpu().float().item())
                domain_data.append(domains[i])

    # 4. Compiling statistics
    metrics_summary = {}
    macro_acc = []
    
    auc_values = []
    valid_auc_count = 0
    undefined_auc_count = 0
    
    print("\n" + "=" * 50)
    print("QWENQURATER EVALUATION SUMMARY")
    print("=" * 50)
    
    for dim_idx, dim_name in enumerate(DIMENSION_NAMES):
        targets = np.array(all_targets[dim_idx])
        preds = np.array(all_preds[dim_idx])
        diffs = np.array(all_diffs[dim_idx])
        
        if len(targets) == 0:
            continue
            
        bce_loss = float(-np.mean(targets * np.log(np.clip(preds, 1e-7, 1-1e-7)) + (1.0 - targets) * np.log(np.clip(1.0 - preds, 1e-7, 1-1e-7))))
        
        pred_label = (preds > 0.5).astype(int)
        gt_label = (targets > 0.5).astype(int)
        
        accuracy = float(np.mean(pred_label == gt_label))
        macro_acc.append(accuracy)
        
        balanced_acc = compute_balanced_accuracy(gt_label, pred_label)
        
        # Safe AUC Calculation
        auc_res = compute_auc(targets, preds)
        if auc_res["auc_status"] == "OK":
            valid_auc_count += 1
            auc_values.append(auc_res["auc"])
        else:
            undefined_auc_count += 1
            
        # Calculate distributions
        diff_stats = get_distribution_stats(diffs)
        prob_stats = get_distribution_stats(preds)
        buckets = compute_confidence_buckets(targets, preds)
        
        print(f"Dimension: {dim_name}")
        auc_val = auc_res["auc"]
        auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A (Single Class)"
        
        print(f"  Accuracy          : {accuracy:.4f}")
        print(f"  Balanced Accuracy : {f'{balanced_acc:.4f}' if balanced_acc is not None else 'N/A'}")
        print(f"  BCE Loss          : {bce_loss:.4f}")
        print(f"  AUC Score         : {auc_str}")
        print(f"  Confidence Buckets Acc:")
        print(f"    - Low (<0.3)    : {buckets['low']}")
        print(f"    - Medium        : {buckets['medium']}")
        print(f"    - High (>=0.7)  : {buckets['high']}")
        print(f"  Score Diff (B-A)  : mean={diff_stats.get('mean', 0.0):.4f}, std={diff_stats.get('std', 0.0):.4f}")
        print(f"  Prediction Prob   : mean={prob_stats.get('mean', 0.0):.4f}, std={prob_stats.get('std', 0.0):.4f}")
        print("-" * 50)
        
        metrics_summary[dim_name] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "bce_loss": bce_loss,
            "auc": auc_res["auc"],
            "auc_status": auc_res["auc_status"],
            "positive_count": auc_res["positive_count"],
            "negative_count": auc_res["negative_count"],
            "confidence_buckets": buckets,
            "score_diff_distribution": diff_stats,
            "prediction_probability_distribution": prob_stats
        }

    mean_macro_acc = float(np.mean(macro_acc)) if macro_acc else 0.0
    mean_macro_auc = float(np.mean(auc_values)) if auc_values else None
    
    print(f"Macro Accuracy across Dimensions: {mean_macro_acc:.4f}")
    print(f"Macro AUC across Dimensions     : {f'{mean_macro_auc:.4f}' if mean_macro_auc is not None else 'N/A'}")
    
    metrics_summary["macro_accuracy"] = mean_macro_acc
    metrics_summary["macro_auc"] = mean_macro_auc
    metrics_summary["valid_auc_dimension_count"] = valid_auc_count
    metrics_summary["undefined_auc_dimension_count"] = undefined_auc_count

    # 5. Populate verification metadata fields
    metrics_summary["checkpoint_path"] = args.checkpoint_dir
    metrics_summary["base_model_path"] = args.model_path
    
    # Calculate sha256 for rating head
    heads_path_safetensors = os.path.join(args.checkpoint_dir, "rating_head.safetensors")
    if os.path.exists(heads_path_safetensors):
        metrics_summary["rating_head_file_sha256"] = get_sha256(heads_path_safetensors)
    else:
        heads_path_pt = os.path.join(args.checkpoint_dir, "rating_head.pt")
        if os.path.exists(heads_path_pt):
            metrics_summary["rating_head_file_sha256"] = get_sha256(heads_path_pt)
        else:
            print("[CRITICAL ERROR] Rating head weight file is missing in checkpoint!")
            sys.exit(1)
            
    # Calculate sha256 for adapter
    if use_lora:
        adapter_bin = os.path.join(args.checkpoint_dir, "adapter", "adapter_model.bin")
        if not os.path.exists(adapter_bin):
            adapter_bin = os.path.join(args.checkpoint_dir, "adapter", "adapter_model.safetensors")
            
        if os.path.exists(adapter_bin):
            metrics_summary["adapter_file_sha256"] = get_sha256(adapter_bin)
        else:
            backbone_pt = os.path.join(args.checkpoint_dir, "adapter", "backbone.pt")
            if os.path.exists(backbone_pt):
                metrics_summary["adapter_file_sha256"] = get_sha256(backbone_pt)
            else:
                print("[CRITICAL ERROR] Backbone/Adapter weight file is missing in checkpoint!")
                sys.exit(1)
    else:
        metrics_summary["adapter_file_sha256"] = "N/A"
            
    metrics_summary["seed"] = q_config.get("seed", 42)

    # Leakage/Imbalance warnings
    if mean_macro_acc > 0.85:
        print("\n" + "!" * 80)
        print("WARNING: Possible data leakage, label imbalance, duplicated pairs,")
        print("         incorrect label direction, or evaluation bug.")
        print("!" * 80 + "\n")
        
    # Recursively clean NaNs and Infinities to None
    cleaned_metrics = clean_nan_inf(metrics_summary)
    
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_metrics, f, indent=2)
        print(f"\nSaved cleaned metrics summary to: {args.output_file}")
        
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_evaluation()
