from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Any

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from data.qurating_dataset import NormalizedPairwiseDataset

def compute_auc(y_true: List[float], y_scores: List[float]) -> float:
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    pos_labels = (y_true > 0.5).astype(int)
    if len(np.unique(pos_labels)) < 2:
        return 0.5
        
    pos_count = np.sum(pos_labels)
    neg_count = len(pos_labels) - pos_count
    
    sorted_indices = np.argsort(y_scores)
    pos_labels_sorted = pos_labels[sorted_indices]
    
    ranks = np.arange(1, len(pos_labels) + 1)
    pos_ranks_sum = np.sum(ranks * pos_labels_sorted)
    
    auc = (pos_ranks_sum - (pos_count * (pos_count + 1)) / 2) / (pos_count * neg_count)
    return float(auc)

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    tp_rate = np.mean(y_pred[pos_mask] == 1) if np.sum(pos_mask) > 0 else 0.0
    tn_rate = np.mean(y_pred[neg_mask] == 0) if np.sum(neg_mask) > 0 else 0.0
    
    return float(0.5 * (tp_rate + tn_rate))

def compute_confidence_buckets(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float | str]:
    confidences = 2.0 * np.abs(targets - 0.5)
    
    low_mask = (confidences < 0.3)
    med_mask = (confidences >= 0.3) & (confidences < 0.7)
    high_mask = (confidences >= 0.7)
    
    def get_acc(mask):
        if np.sum(mask) == 0:
            return "N/A"
        pred_label = (preds[mask] > 0.5).astype(int)
        gt_label = (targets[mask] > 0.5).astype(int)
        return float(np.mean(pred_label == gt_label))
        
    return {
        "low": get_acc(low_mask),
        "medium": get_acc(med_mask),
        "high": get_acc(high_mask)
    }

def get_distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }

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
    config_path = os.path.join(args.checkpoint_dir, "qurater_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            q_config = json.load(f)
        if q_config.get("use_lora", False) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {adapter_dir} ...")
            model.backbone = PeftModel.from_pretrained(model.backbone, adapter_dir)
            
    # Load scalar rating head
    heads_path = os.path.join(args.checkpoint_dir, "rating_head.safetensors")
    if os.path.exists(heads_path):
        from safetensors.torch import load_file
        model.score.load_state_dict(load_file(heads_path, map_location=device))
    else:
        pt_path = os.path.join(args.checkpoint_dir, "rating_head.pt")
        model.score.load_state_dict(torch.load(pt_path, map_location=device))
        
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
            
            # Forward: model returns (batch_size, 4)
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
        auc_score = compute_auc(targets, preds)
        
        # Calculate distributions
        diff_stats = get_distribution_stats(diffs)
        prob_stats = get_distribution_stats(preds)
        buckets = compute_confidence_buckets(targets, preds)
        
        print(f"Dimension: {dim_name}")
        print(f"  Accuracy          : {accuracy:.4f}")
        print(f"  Balanced Accuracy : {balanced_acc:.4f}")
        print(f"  BCE Loss          : {bce_loss:.4f}")
        print(f"  AUC Score         : {auc_score:.4f}")
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
            "auc": auc_score,
            "confidence_buckets": buckets,
            "score_diff_distribution": diff_stats,
            "prediction_probability_distribution": prob_stats
        }

    mean_macro_acc = float(np.mean(macro_acc)) if macro_acc else 0.0
    print(f"Macro Accuracy across Dimensions: {mean_macro_acc:.4f}")
    metrics_summary["macro_accuracy"] = mean_macro_acc

    # Leakage/Imbalance warnings
    if mean_macro_acc > 0.85:
        print("\n" + "!" * 80)
        print("WARNING: Possible data leakage, label imbalance, duplicated pairs,")
        print("         incorrect label direction, or evaluation bug.")
        print("!" * 80 + "\n")
        
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nSaved metrics summary to: {args.output_file}")
        
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_evaluation()
