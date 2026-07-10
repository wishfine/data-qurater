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
    
    # 1. Save LoRA adapter (under adapter/)
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    if args.use_lora:
        model.backbone.save_pretrained(adapter_dir)
    else:
        os.makedirs(adapter_dir, exist_ok=True)
        torch.save(model.backbone.state_dict(), os.path.join(adapter_dir, "backbone.pt"))
        
    # 2. Save rating head (rating_head.safetensors)
    heads_path = os.path.join(checkpoint_dir, "rating_head.safetensors")
    try:
        from safetensors.torch import save_file
        save_file(model.score.state_dict(), heads_path)
    except ImportError:
        torch.save(model.score.state_dict(), os.path.join(checkpoint_dir, "rating_head.pt"))
        
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
        
    model.to(device=device, dtype=dtype)
    
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
        config = model.backbone.config
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=train_dataset.collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_train_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    start_epoch = 0
    if args.resume_from_checkpoint:
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(os.path.join(args.resume_from_checkpoint, "trainer_state.pt"), map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1

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
                    head_grad_norm = sum(p.grad.norm().item() for p in model.score.parameters() if p.grad is not None)
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
            
        if stop_training:
            if is_main_process:
                print(f"Reached max_optimizer_steps: {args.max_optimizer_steps}. Stopping training.")
            break
            
    if is_main_process:
        final_dir = os.path.join(args.output_dir, "checkpoint-final")
        save_modular_checkpoint(model, tokenizer, final_dir, args, args.num_train_epochs, target_modules, optimizer, scheduler)

if __name__ == "__main__":
    main()
