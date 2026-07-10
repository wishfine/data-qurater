from __future__ import annotations
import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from data.qurating_dataset import NormalizedPairwiseDataset, OfficialQuRatingDatasetAdapter

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
    """
    Bradley-Terry Pairwise Loss with BCE With Logits.
    logit = s_B - s_A
    P(B > A) = sigmoid(logit)
    Handles zero-valid masks gracefully to prevent NaN.
    """
    logits = ratings_b.float() - ratings_a.float()
    
    # Calculate unreduced pairwise loss
    per_sample_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    
    # Filter by confidence threshold
    valid_mask = (confidences >= confidence_threshold)
    num_valid = valid_mask.sum()
    
    if num_valid == 0:
        # Return a zero loss attached to the gradient graph to avoid NaNs
        return 0.0 * logits.sum()
        
    return per_sample_loss[valid_mask].mean()

def save_modular_checkpoint(model, tokenizer, checkpoint_dir, args, epoch, optimizer=None, scheduler=None):
    """
    Save checkpoints in the mandated directory format:
    checkpoint-epoch-{epoch}/
    ├── adapter/
    ├── rating_head.safetensors
    ├── qurater_config.json
    ├── tokenizer/
    ├── training_args.json
    └── trainer_state.pt
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Save LoRA adapter (under adapter/)
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    if args.use_lora:
        # PEFT save_pretrained saves adapter_config.json and adapter_model.bin/safetensors
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
        "model_path": args.model_path,
        "pooling_type": "last_token",
        "use_lora": args.use_lora,
        "use_4bit": args.use_4bit,
        "dimension_mapping": DIMENSION_NAMES
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

def inspect_qwen_features(model_path: str, model: nn.Module):
    """Inspect and print Hybrid Linear Attention / Gated DeltaNet status for Qwen models"""
    print("\n" + "=" * 50)
    print("QWEN3.5 ATTENTION & PATHWAY CONFIGURATION CHECK")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
    config = getattr(model, "config", None)
    if config is not None:
        print(f"Model Type: {getattr(config, 'model_type', 'Unknown')}")
        
    has_gated_deltanet = False
    has_linear_attn = False
    for name, module in model.named_modules():
        name_lower = name.lower()
        if "deltanet" in name_lower or "gated_delta" in name_lower:
            has_gated_deltanet = True
        if "linearattention" in name_lower or "linear_attn" in name_lower:
            has_linear_attn = True
            
    print(f"Gated DeltaNet Modules Detected: {has_gated_deltanet}")
    print(f"Hybrid Linear Attention Modules Detected: {has_linear_attn}")
    
    try:
        import fla
        print("fla library: Successfully imported. Fast path is AVAILABLE.")
    except ImportError:
        print("fla library: Not found. Linear attention fast path is UNAVAILABLE (will use torch fallback).")
    print("=" * 50 + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train QwenQuRater with Bradley-Terry Pairwise Loss")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base Qwen3.5-4B model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to pairwise training json/jsonl")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to pairwise validation json/jsonl")
    parser.add_argument("--output_dir", type=str, default="./qurater_output", help="Directory to save model checkpoints")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--use_4bit", action="store_true", help="Quantize backbone model to 4-bit NF4")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold filter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
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
        print(f"Loading base model from: {args.model_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"  # Fixed padding side to right
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Configure rank-specific device mapping for 4-bit loading compatibility
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
        
    try:
        backbone = AutoModel.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load model {args.model_path}: {e}")
        sys.exit(1)
        
    inspect_qwen_features(args.model_path, backbone)
    
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        
    model = QwenQuRater(backbone=backbone)
    
    # Prepare PEFT config
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.use_4bit:
            model.backbone = prepare_model_for_kbit_training(model.backbone)
            
        # Dynamically scan linear layers in both standard self-attention and DeltaNet linear attention modules
        target_modules = []
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name.split(".")[-1])
        target_modules = list(set(target_modules))
        
        # Enforce error if no targets match to prevent silent failures
        if not target_modules:
            raise ValueError("[CRITICAL ERROR] No target modules matched for LoRA training! Verify model definition.")
            
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        model.backbone = get_peft_model(model.backbone, lora_config)
        
    model.to(device)
    
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
        print("\n" + "=" * 50)
        print("OPTIMIZER PARAMETER GROUPS VERIFICATION")
        print("=" * 50)
        print(f"Total Trainable LoRA Parameters: {len(lora_params)}")
        print(f"Total Trainable Scalar Head Parameters: {len(head_params)}")
        print("=" * 50 + "\n")
        
    # Enforce gradient constraints on target server
    if args.use_lora:
        assert len(lora_params) > 0, "ERROR: No LoRA parameters are set as trainable!"
    assert len(head_params) > 0, "ERROR: Joint rating head (score) parameters are not trainable!"

    # 4. Load datasets
    train_dataset = NormalizedPairwiseDataset(args.train_file, tokenizer, args.max_length, args.max_train_samples)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = None
    if args.validation_file:
        val_dataset = NormalizedPairwiseDataset(args.validation_file, tokenizer, args.max_length, args.max_eval_samples)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=val_dataset.collate_fn
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

    # 5. Training Loop with Benchmark
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
            step_start = time.time()
            
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            targets = batch["targets"].to(device)
            dimension_ids = batch["dimension_ids"].to(device)
            confidences = batch["confidences"].to(device)
            
            # Forward pass: extract score for specific dimension_id
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
            if (step + 1) % args.gradient_accumulation_steps == 0:
                opt_start = time.time()
                
                # Check gradients
                if step < 10 and is_main_process:
                    head_grad_norm = sum(p.grad.norm().item() for p in model.score.parameters() if p.grad is not None)
                    print(f"  [GRAD VERIFY] Joint Rating Head Gradient Norm: {head_grad_norm:.6f}")
                    assert head_grad_norm > 0.0 or epoch > 0, "ERROR: Rating heads have zero gradient!"
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_time = time.time() - opt_start
                optimizer_times.append(opt_time)
                
            step_times.append(time.time() - step_start)
            epoch_loss += batch_loss.item() * args.gradient_accumulation_steps
            
            # 10-step benchmark output
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
                    
        # Epoch metrics & save modular checkpoint
        avg_loss = epoch_loss / len(train_loader)
        if is_main_process:
            print(f"\nEpoch {epoch+1} Complete. Avg Pairwise loss: {avg_loss:.4f}")
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            save_modular_checkpoint(model, tokenizer, checkpoint_dir, args, epoch, optimizer, scheduler)
            
    if is_main_process:
        final_dir = os.path.join(args.output_dir, "checkpoint-final")
        save_modular_checkpoint(model, tokenizer, final_dir, args, args.num_train_epochs, optimizer, scheduler)

if __name__ == "__main__":
    main()
