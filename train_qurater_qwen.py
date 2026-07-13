from __future__ import annotations
import os
import sys
import json
import time
import argparse
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from data.qurating_dataset import NormalizedPairwiseDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def bradley_terry_loss(
    ratings_a: torch.Tensor, 
    ratings_b: torch.Tensor, 
    targets: torch.Tensor,
    confidences: torch.Tensor,
    confidence_threshold: float = 0.0
) -> torch.Tensor:
    logits = ratings_b.float() - ratings_a.float()
    
    # Calculate unreduced pairwise loss
    per_sample_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    
    # Filter by confidence threshold
    valid_mask = (confidences >= confidence_threshold)
    num_valid = valid_mask.sum()
    
    if num_valid == 0:
        return 0.0 * logits.sum()
        
    return per_sample_loss[valid_mask].mean()

def save_modular_checkpoint(model, tokenizer, checkpoint_dir, args, epoch, target_modules=None, optimizer=None, scheduler=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Unwrap DDP model if necessary
    raw_model = model.module if hasattr(model, "module") else model
    
    # 1. Save LoRA adapter (under adapter/)
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    if args.use_lora:
        raw_model.backbone.save_pretrained(adapter_dir)
    else:
        os.makedirs(adapter_dir, exist_ok=True)
        torch.save(raw_model.backbone.state_dict(), os.path.join(adapter_dir, "backbone.pt"))
        
    # 2. Save rating head (rating_head.safetensors)
    heads_path = os.path.join(checkpoint_dir, "rating_head.safetensors")
    try:
        from safetensors.torch import save_file
        save_file(raw_model.score.state_dict(), heads_path)
    except ImportError:
        torch.save(raw_model.score.state_dict(), os.path.join(checkpoint_dir, "rating_head.pt"))
        
    # 3. Save tokenizer (tokenizer/)
    tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    
    # 4. Save qurater_config.json
    q_config = {
        "model_id": "Qwen/Qwen3.5-4B",
        "base_model_path": args.model_path,
        "dimensions": DIMENSION_NAMES,
        "pooling": "last_non_padding",
        "padding_side": "right",
        "use_4bit": args.use_4bit,
        "use_lora": args.use_lora,
        "lora_target_modules": target_modules if target_modules else [],
        "max_length": args.max_length,
        "torch_dtype": "bfloat16" if args.bf16 else "float32",
        "seed": args.seed
    }
    with open(os.path.join(checkpoint_dir, "qurater_config.json"), "w", encoding="utf-8") as f:
        json.dump(q_config, f, indent=2)
        
    # 5. Save training_args.json
    with open(os.path.join(checkpoint_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
        
    # 6. Save trainer_state.pt
    state = {
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))
    print(f"[CHECKPOINT] Saved modular checkpoint directory to: {checkpoint_dir}")

def save_experiment_metadata(args, val_dataset):
    os.makedirs("outputs/qwen35_4b_experiment", exist_ok=True)
    
    # Helper to calculate file SHA256
    def get_sha256(path):
        if not os.path.exists(path):
            return "N/A"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
        
    val_file_path = args.validation_file if args.validation_file else ""
    val_sha = get_sha256(val_file_path) if val_file_path else "N/A"
    
    # 1. Save eval_manifest.json
    manifest = {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_path": args.model_path,
        "validation_file": val_file_path,
        "validation_file_sha256": val_sha,
        "num_samples": len(val_dataset) if val_dataset else 0,
        "max_length": args.max_length,
        "seed": args.seed,
        "confidence_threshold": args.confidence_threshold,
        "dimensions": DIMENSION_NAMES
    }
    with open("outputs/qwen35_4b_experiment/eval_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # 2. Save experiment_config.json
    with open("outputs/qwen35_4b_experiment/experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

def evaluate_model(model, val_dataset, device, args, epoch_name):
    model.eval()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    from evaluate_qurater import (
        compute_auc, 
        compute_balanced_accuracy, 
        get_distribution_stats, 
        compute_confidence_buckets
    )
    
    all_targets = {dim_idx: [] for dim_idx in range(4)}
    all_preds = {dim_idx: [] for dim_idx in range(4)}
    all_diffs = {dim_idx: [] for dim_idx in range(4)}
    all_confidences = {dim_idx: [] for dim_idx in range(4)}
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            targets = batch["targets"].to(device)
            dimension_ids = batch["dimension_ids"].to(device)
            confidences = batch["confidences"].to(device)
            
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

    metrics_summary = {}
    macro_acc = []
    valid_auc_count = 0
    auc_values = []
    
    global_rank = int(os.environ.get("RANK", 0))
    is_main = (global_rank == 0)
    
    if is_main:
        print("\n" + "=" * 50)
        print(f"VAL EVALUATION SUMMARY AT {epoch_name}")
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
        auc_res = compute_auc(targets, preds)
        if auc_res["auc_status"] == "OK":
            valid_auc_count += 1
            auc_values.append(auc_res["auc"])
            
        diff_stats = get_distribution_stats(diffs)
        prob_stats = get_distribution_stats(preds)
        buckets = compute_confidence_buckets(targets, preds)
        
        auc_val = auc_res["auc"]
        auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
        
        if is_main:
            print(f"Dimension: {dim_name}")
            print(f"  Accuracy          : {accuracy:.4f}")
            print(f"  Balanced Accuracy : {f'{balanced_acc:.4f}' if balanced_acc is not None else 'N/A'}")
            print(f"  BCE Loss          : {bce_loss:.4f}")
            print(f"  AUC Score         : {auc_str}")
        
        metrics_summary[dim_name] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "bce_loss": bce_loss,
            "auc": auc_val,
            "score_diff_distribution": diff_stats,
            "prob_distribution": prob_stats,
            "confidence_buckets": buckets
        }
        
    macro_accuracy_val = float(np.mean(macro_acc)) if macro_acc else 0.0
    macro_auc_val = float(np.mean(auc_values)) if valid_auc_count > 0 else None
    
    if is_main:
        print("-" * 50)
        print(f"Macro Accuracy across Dimensions: {macro_accuracy_val:.4f}")
        print(f"Macro AUC across Dimensions     : {f'{macro_auc_val:.4f}' if macro_auc_val is not None else 'N/A'}")
        print("=" * 50 + "\n")
    
    metrics_summary["macro_accuracy"] = macro_accuracy_val
    metrics_summary["macro_auc"] = macro_auc_val
    
    if is_main:
        eval_dir = os.path.join(os.path.dirname(args.output_dir), "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        eval_file_path = os.path.join(eval_dir, f"{epoch_name}_eval.json")
        with open(eval_file_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"[EVALUATION] Saved metrics to: {eval_file_path}")
    
    model.train()
    return metrics_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Train QwenQuRater with Bradley-Terry Pairwise Loss")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base Qwen3.5-4B model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to pairwise training json/jsonl")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to pairwise validation json/jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen35_4b_experiment/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--use_4bit", action="store_true", default=False, help="Quantize backbone model to 4-bit NF4")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold filter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_optimizer_steps", type=int, default=None, help="Stop training after reaching this number of optimizer steps")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Distributed Setup
    if "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_distributed = True
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_distributed = False
        
    is_main_process = (global_rank == 0)
    
    # 2. Load Model and Tokenizer
    if is_main_process:
        print(f"Loading Qwen3.5-4B base model from: {args.model_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"  # Fixed padding side to right
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device_map = None
    if torch.cuda.is_available():
        device_map = {"": local_rank}
        
    bnb_config = None
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float32
    
    try:
        backbone = AutoModel.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load model {args.model_path}: {e}")
        sys.exit(1)
        
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        
    model = QwenQuRater(backbone=backbone)
    
    # Prepare PEFT LoRA config
    target_modules = []
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.use_4bit:
            model.backbone = prepare_model_for_kbit_training(model.backbone)
            
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name.split(".")[-1])
        target_modules = list(set(target_modules))
        
        # Enforce error if no targets match to prevent silent failures
        if not target_modules:
            raise RuntimeError("[CRITICAL ERROR] No target modules matched for LoRA training! Verify model definition.")
            
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        model.backbone = get_peft_model(model.backbone, lora_config)
        
    if device_map is not None:
        model.score.to(device=device, dtype=dtype)
    else:
        model.to(device=device, dtype=dtype)
        
    # Load model weights if resuming from checkpoint (must do before DDP wrapping)
    if args.resume_from_checkpoint:
        if is_main_process:
            print(f"Loading model weights from checkpoint: {args.resume_from_checkpoint}")
        
        # 1. Load LoRA adapter weights
        adapter_dir = os.path.join(args.resume_from_checkpoint, "adapter")
        if args.use_lora:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
            bin_path = os.path.join(adapter_dir, "adapter_model.bin")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path, device=device)
            else:
                state_dict = torch.load(bin_path, map_location=device)
            set_peft_model_state_dict(model.backbone, state_dict)
        else:
            backbone_path = os.path.join(adapter_dir, "backbone.pt")
            if os.path.exists(backbone_path):
                model.backbone.load_state_dict(torch.load(backbone_path, map_location=device))
            
        # 2. Load rating head weights
        heads_path = os.path.join(args.resume_from_checkpoint, "rating_head.safetensors")
        heads_bin_path = os.path.join(args.resume_from_checkpoint, "rating_head.pt")
        if os.path.exists(heads_path):
            from safetensors.torch import load_file
            head_state = load_file(heads_path, device=device)
            model.score.load_state_dict(head_state)
        elif os.path.exists(heads_bin_path):
            head_state = torch.load(heads_bin_path, map_location=device)
            model.score.load_state_dict(head_state)
            
    # Wrap in DistributedDataParallel for multi-GPU training
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
    # Calculate parameter counts and ratios
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_trainable_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and "lora_" in name)
    head_trainable_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and "score" in name)
    
    # 3. Verify Optimizer Parameter Groups (requires both LoRA and Rating Head to be trainable)
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                lora_params.append((name, param))
            elif "score" in name:
                head_params.append((name, param))
                
    if is_main_process:
        raw_model = model.module if hasattr(model, "module") else model
        config = raw_model.backbone.config
        if hasattr(config, "text_config"):
            config = config.text_config
        hidden_size = getattr(config, "hidden_size", getattr(config, "hidden_dim", "Unknown"))
        num_layers = getattr(config, "num_hidden_layers", "Unknown")
        
        print("\n" + "=" * 50)
        print("OPTIMIZER PARAMETER GROUPS VERIFICATION")
        print("=" * 50)
        print(f"Model ID                  : Qwen/Qwen3.5-4B")
        print(f"Model Hidden Size         : {hidden_size}")
        print(f"Backbone Layer Count      : {num_layers}")
        if args.use_lora:
            print(f"LoRA Target Modules       : {target_modules}")
        print(f"Total Trainable LoRA Params: {lora_trainable_params}")
        print(f"Total Trainable Head Params: {head_trainable_params}")
        print(f"Total Model Parameters    : {total_params}")
        print(f"Total Trainable Parameters: {trainable_params}")
        print(f"Trainable Ratio           : {100 * trainable_params / total_params:.4f}%")
        print("=" * 50 + "\n")
        
    # Enforce gradient constraints on target server
    if args.use_lora:
        assert len(lora_params) > 0, "ERROR: No LoRA parameters are set as trainable!"
    assert len(head_params) > 0, "ERROR: Joint rating head (score) parameters are not trainable!"

    # 4. Load datasets
    train_dataset = NormalizedPairwiseDataset(args.train_file, tokenizer, args.max_length, args.max_train_samples)
    
    val_dataset = None
    if args.validation_file:
        val_dataset = NormalizedPairwiseDataset(args.validation_file, tokenizer, args.max_length, args.max_eval_samples)
        
    # Save experiment configs and validation manifest
    if is_main_process:
        save_experiment_metadata(args, val_dataset)

    # 5. Save untrained checkpoint-0 baseline BEFORE any optimizer steps
    checkpoint_0_dir = "outputs/qwen35_4b_experiment/checkpoint-0"
    if is_main_process:
        save_modular_checkpoint(model, tokenizer, checkpoint_0_dir, args, epoch=0, target_modules=target_modules)
        if val_dataset is not None:
            evaluate_model(model, val_dataset, device, args, "epoch_0.0")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_train_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    start_epoch = 0
    if args.resume_from_checkpoint:
        if is_main_process:
            print(f"Resuming optimizer/scheduler state from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(os.path.join(args.resume_from_checkpoint, "trainer_state.pt"), map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])

    # 6. Training Loop
    optimizer_step = 0
    micro_step = 0
    stop_training = False
    
    for epoch in range(start_epoch, args.num_train_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0.0
        
        step_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []
        
        for step, batch in enumerate(train_loader):
            micro_step += 1
            step_start = time.time()
            
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            targets = batch["targets"].to(device)
            dimension_ids = batch["dimension_ids"].to(device)
            confidences = batch["confidences"].to(device)
            
            # Forward pass
            f_start = time.time()
            ratings_a = model(input_ids_a, attention_mask_a, dimension_ids)
            ratings_b = model(input_ids_b, attention_mask_b, dimension_ids)
            
            batch_loss = bradley_terry_loss(
                ratings_a, 
                ratings_b, 
                targets, 
                confidences, 
                args.confidence_threshold
            )
            batch_loss = batch_loss / args.gradient_accumulation_steps
            forward_times.append(time.time() - f_start)
            
            # Backward pass
            b_start = time.time()
            batch_loss.backward()
            backward_times.append(time.time() - b_start)
            
            # Optimizer step
            opt_time = 0.0
            if micro_step % args.gradient_accumulation_steps == 0:
                opt_start = time.time()
                
                # Check gradients on target server
                if optimizer_step < 10 and is_main_process:
                    raw_model = model.module if hasattr(model, "module") else model
                    head_grad_norm = sum(p.grad.norm().item() for p in raw_model.score.parameters() if p.grad is not None)
                    lora_grad_norm = sum(p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None and "lora_" in name)
                    
                    print(f"  [GRAD VERIFY] Joint Rating Head Gradient Norm: {head_grad_norm:.6f}")
                    print(f"  [GRAD VERIFY] LoRA Gradient Norm             : {lora_grad_norm:.6f}")
                    
                    assert torch.isfinite(batch_loss), "Loss is NaN/Inf!"
                    assert head_grad_norm > 0.0 or epoch > 0, "ERROR: Rating heads have zero gradient!"
                    if args.use_lora:
                        assert lora_grad_norm > 0.0 or epoch > 0, "ERROR: LoRA weights have zero gradient!"
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                optimizer_step += 1
                opt_time = time.time() - opt_start
                optimizer_times.append(opt_time)
                
                if is_main_process:
                    print(f"  [STEP] micro_step: {micro_step} | optimizer_step: {optimizer_step}")
                
                # Check for 0.25-epoch intervals (excluding full epochs, which are handled at the end of the epoch loop)
                steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
                quarter_epoch_steps = max(1, steps_per_epoch // 4)
                if optimizer_step > 0 and (optimizer_step % quarter_epoch_steps == 0) and (optimizer_step % steps_per_epoch != 0):
                    epoch_decimal = optimizer_step / steps_per_epoch
                    epoch_name = f"epoch_{epoch_decimal:.2f}"
                    if is_main_process:
                        print(f"\n[INTERVAL] Reached {epoch_name} (step {optimizer_step}). Saving checkpoint and evaluating...")
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch_name.replace('_', '-')}")
                        save_modular_checkpoint(model, tokenizer, checkpoint_dir, args, epoch_decimal, target_modules, optimizer, scheduler)
                        if val_dataset is not None:
                            evaluate_model(model, val_dataset, device, args, epoch_name)
                            
                # Periodic safety checkpoints every 5000 steps to prevent loss of progress
                if optimizer_step > 0 and (optimizer_step % 5000 == 0):
                    epoch_decimal = optimizer_step / steps_per_epoch
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-step-{optimizer_step}")
                    if is_main_process:
                        print(f"\n[SAFETY CHECKPOINT] Reached step {optimizer_step}. Saving recovery checkpoint to {checkpoint_dir}...")
                        save_modular_checkpoint(model, tokenizer, checkpoint_dir, args, epoch_decimal, target_modules, optimizer, scheduler)
                    
                if args.max_optimizer_steps is not None and optimizer_step >= args.max_optimizer_steps:
                    stop_training = True
                    break
                
            step_times.append(time.time() - step_start)
            epoch_loss += batch_loss.item() * args.gradient_accumulation_steps
            
            # Benchmark outputs
            if step < 10 and is_main_process:
                tokens_per_sec = (input_ids_a.numel() + input_ids_b.numel()) / step_times[-1]
                print(f"[BENCHMARK STEP {step+1}]")
                print(f"  Step Latency: {step_times[-1]:.4f}s")
                print(f"  Forward Latency: {forward_times[-1]:.4f}s")
                print(f"  Backward Latency: {backward_times[-1]:.4f}s")
                if opt_time > 0:
                    print(f"  Optimizer step Latency: {opt_time:.4f}s")
                print(f"  Throughput: {tokens_per_sec:.2f} tokens/s")
                if torch.cuda.is_available():
                    print(f"  GPU Max Memory Allocated: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")
                    
        # Epoch checkpoint
        avg_loss = epoch_loss / len(train_loader)
        if is_main_process:
            print(f"\nEpoch {epoch+1} Complete. Avg Pairwise loss: {avg_loss:.4f}")
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            save_modular_checkpoint(model, tokenizer, checkpoint_dir, args, epoch+1, target_modules, optimizer, scheduler)
            if val_dataset is not None:
                evaluate_model(model, val_dataset, device, args, f"epoch_{float(epoch+1)}")
            
        if stop_training:
            if is_main_process:
                print(f"Reached max_optimizer_steps: {args.max_optimizer_steps}. Stopping training.")
            break
            
    if is_main_process:
        final_dir = os.path.join(args.output_dir, "checkpoint-final")
        save_modular_checkpoint(model, tokenizer, final_dir, args, args.num_train_epochs, target_modules, optimizer, scheduler)

if __name__ == "__main__":
    main()
