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

from models.qwen_qurater import QwenQuRater, QUALITY_DIMENSIONS
from data.qurating_dataset import PairwiseDataset

def compute_auc(y_true: List[float], y_scores: List[float]) -> float:
    """Compute Area Under ROC Curve using rank sums (Wilcoxon-Mann-Whitney formula)"""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Binarize ground truth labels around 0.5
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

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate QwenQuRater quality predictor")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model directory")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to modular checkpoint directory")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--output_file", type=str, default=None, help="Save evaluation metrics summary to this json file")
    parser.add_argument("--pooling_type", type=str, default="last_token", help="Pooling strategy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    model = QwenQuRater(
        backbone=backbone,
        pooling_type=args.pooling_type,
        padding_side=tokenizer.padding_side
    )
    
    # Load LoRA adapter if use_lora is true in metadata config
    config_path = os.path.join(args.checkpoint_dir, "qurater_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            q_config = json.load(f)
        if q_config.get("use_lora", False):
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {args.checkpoint_dir} ...")
            model.backbone = PeftModel.from_pretrained(model.backbone, args.checkpoint_dir)
            
    # Load scalar heads
    heads_path = os.path.join(args.checkpoint_dir, "rating_heads.pt")
    model.rating_heads.load_state_dict(torch.load(heads_path, map_location=device))
    
    model.to(device)
    model.eval()

    # 2. Load Evaluation Dataset
    print(f"Loading dataset from: {args.eval_file}")
    dataset = PairwiseDataset(args.eval_file, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # 3. Running evaluation predictions
    all_targets = {dim: [] for dim in QUALITY_DIMENSIONS}
    all_preds = {dim: [] for dim in QUALITY_DIMENSIONS}
    all_diffs = {dim: [] for dim in QUALITY_DIMENSIONS}
    swap_errors = {dim: [] for dim in QUALITY_DIMENSIONS}
    
    domain_data = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                domain_data.append(item.get("domain", "general"))

    print("Evaluating samples...")
    idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            prob_labels = {k: v.to(device) for k, v in batch["prob_labels"].items()}
            
            # Predict original (A, B)
            ratings_a = model(input_ids_a, attention_mask_a)
            ratings_b = model(input_ids_b, attention_mask_b)
            
            # Predict swapped (B, A)
            ratings_a_swap = model(input_ids_b, attention_mask_b)
            ratings_b_swap = model(input_ids_a, attention_mask_a)
            
            batch_size = input_ids_a.size(0)
            
            for dim in QUALITY_DIMENSIONS:
                r_a = ratings_a[dim].cpu().float().numpy()
                r_b = ratings_b[dim].cpu().float().numpy()
                gt = prob_labels[dim].cpu().float().numpy()
                
                logits = r_b - r_a
                p_pred = 1.0 / (1.0 + np.exp(-logits))
                
                r_a_swap = ratings_a_swap[dim].cpu().float().numpy()
                r_b_swap = ratings_b_swap[dim].cpu().float().numpy()
                logits_swap = r_b_swap - r_a_swap
                p_pred_swap = 1.0 / (1.0 + np.exp(-logits_swap))
                
                swap_sum = p_pred + p_pred_swap
                swap_err = np.abs(swap_sum - 1.0)
                
                all_targets[dim].extend(gt.tolist())
                all_preds[dim].extend(p_pred.tolist())
                all_diffs[dim].extend(logits.tolist())
                swap_errors[dim].extend(swap_err.tolist())
                
            idx += batch_size

    # 4. Compiling statistics
    metrics_summary = {}
    macro_acc = []
    
    print("\n" + "=" * 50)
    print("QWENQURATER EVALUATION SUMMARY")
    print("=" * 50)
    
    for dim in QUALITY_DIMENSIONS:
        targets = np.array(all_targets[dim])
        preds = np.array(all_preds[dim])
        diffs = np.array(all_diffs[dim])
        swap_errs = np.array(swap_errors[dim])
        
        bce_loss = float(-np.mean(targets * np.log(np.clip(preds, 1e-7, 1-1e-7)) + (1.0 - targets) * np.log(np.clip(1.0 - preds, 1e-7, 1-1e-7))))
        
        pred_label = (preds > 0.5).astype(int)
        gt_label = (targets > 0.5).astype(int)
        accuracy = float(np.mean(pred_label == gt_label))
        macro_acc.append(accuracy)
        
        auc_score = compute_auc(targets, preds)
        confidence = float(np.mean(np.abs(preds - 0.5)) * 2)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        mean_swap_err = float(np.mean(swap_errs))
        
        print(f"Dimension: {dim}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  BCE Loss: {bce_loss:.4f}")
        print(f"  AUC Score: {auc_score:.4f}")
        print(f"  Prediction Confidence: {confidence:.4f}")
        print(f"  Score Diff (B - A): mean={mean_diff:.4f}, std={std_diff:.4f}")
        print(f"  Swap Consistency Error: {mean_swap_err:.4e}")
        print("-" * 50)
        
        metrics_summary[dim] = {
            "accuracy": accuracy,
            "bce_loss": bce_loss,
            "auc": auc_score,
            "confidence": confidence,
            "score_diff_mean": mean_diff,
            "score_diff_std": std_diff,
            "swap_consistency_error": mean_swap_err
        }

    mean_macro_acc = float(np.mean(macro_acc))
    print(f"Macro Accuracy across 4 Dimensions: {mean_macro_acc:.4f}")
    metrics_summary["macro_accuracy"] = mean_macro_acc

    unique_domains = list(set(domain_data))
    domain_accuracies = {}
    
    if len(unique_domains) > 1:
        print("\nDomain Specific Accuracies (Macro):")
        for dom in unique_domains:
            dom_indices = [i for i, d in enumerate(domain_data) if d == dom]
            dom_accs = []
            for dim in QUALITY_DIMENSIONS:
                dim_targets = np.array(all_targets[dim])[dom_indices]
                dim_preds = np.array(all_preds[dim])[dom_indices]
                
                pred_label = (dim_preds > 0.5).astype(int)
                gt_label = (dim_targets > 0.5).astype(int)
                dom_accs.append(np.mean(pred_label == gt_label))
            
            mean_dom_acc = float(np.mean(dom_accs))
            print(f"  {dom}: {mean_dom_acc:.4f}")
            domain_accuracies[dom] = mean_dom_acc
            
        metrics_summary["domain_accuracies"] = domain_accuracies
        
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nSaved metrics summary to: {args.output_file}")
        
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_evaluation()
