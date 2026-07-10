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

from models.qwen_qurater import QwenQuRater, QUALITY_DIMENSIONS
from data.qurating_dataset import PairwiseDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def bradley_terry_loss(ratings_a: torch.Tensor, ratings_b: torch.Tensor, p_b_gt_a: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry Pairwise Loss.
    logit = s_B - s_A
    P(B > A) = sigmoid(logit)
    """
    logits = ratings_b.float() - ratings_a.float()
    return F.binary_cross_entropy_with_logits(logits, p_b_gt_a.float(), reduction="mean")

def test_loss_direction():
    """Verify that s_B > s_A yields smaller loss when target is 1.0 (B > A)"""
    s_a = torch.tensor([1.0])
    s_b = torch.tensor([2.0])
    target = torch.tensor([1.0])
    
    loss_preferred = bradley_terry_loss(s_a, s_b, target)
    loss_non_preferred = bradley_terry_loss(s_b, s_a, target)
    
    assert loss_preferred.item() < loss_non_preferred.item(), "BT Loss direction check failed!"
    print("[VERIFICATION] Bradley-Terry Loss direction check PASSED.")

def save_modular_checkpoint(model, tokenizer, checkpoint_dir, args):
    """
    Save checkpoint in modular directories:
    - LoRA adapter (if enabled)
    - 4 scalar rating heads
    - Tokenizer
    - Configuration metadata
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Save Tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    # 2. Save Backbone (either PEFT LoRA adapter or full weights if unwrapped)
    raw_backbone = model.backbone
    if hasattr(raw_backbone, "save_pretrained"):
        raw_backbone.save_pretrained(checkpoint_dir)
    else:
        torch.save(raw_backbone.state_dict(), os.path.join(checkpoint_dir, "backbone.pt"))
        
    # 3. Save Rating Heads
    torch.save(model.rating_heads.state_dict(), os.path.join(checkpoint_dir, "rating_heads.pt"))
    
    # 4. Save Metadata Config
    q_config = {
        "model_path": args.model_path,
        "pooling_type": model.backbone.config.model_type if hasattr(model, "pooling_type") else "last_token",
        "use_lora": args.use_lora,
        "use_4bit": args.use_4bit,
        "head_type": "A",
        "dimension_mapping": QUALITY_DIMENSIONS
    }
    with open(os.path.join(checkpoint_dir, "qurater_config.json"), "w", encoding="utf-8") as f:
        json.dump(q_config, f, indent=2)
        
    print(f"[CHECKPOINT] Saved modular checkpoint to: {checkpoint_dir}")

def verify_checkpoint_round_trip(model, tokenizer, args, device):
    """Perform optimizer step, save, reload, and verify that scores match within 1e-5"""
    print("\n" + "=" * 50)
    print("RUNNING CHECKPOINT ROUND-TRIP TEST")
    print("=" * 50)
    
    test_dir = "./outputs/temp_roundtrip_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Mock inputs
    input_ids = torch.tensor([[10, 20, 30, 0]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long).to(device)
    
    # 1. Forward before save
    model.eval()
    with torch.no_grad():
        scores_before = model(input_ids, attention_mask)
        
    # 2. Save modular checkpoint
    save_modular_checkpoint(model, tokenizer, test_dir, args)
    
    # 3. Reload from modular checkpoint
    # Re-instantiate model structure
    from peft import PeftModel
    bnb_config = None
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
    backbone_new = AutoModel.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    model_new = QwenQuRater(backbone=backbone_new)
    
    if args.use_lora:
        model_new.backbone = PeftModel.from_pretrained(model_new.backbone, test_dir)
        
    model_new.rating_heads.load_state_dict(torch.load(os.path.join(test_dir, "rating_heads.pt"), map_location=device))
    model_new.to(device)
    model_new.eval()
    
    # 4. Forward after reload
    with torch.no_grad():
        scores_after = model_new(input_ids, attention_mask)
        
    # 5. Check precision matching
    for dim in QUALITY_DIMENSIONS:
        diff = torch.max(torch.abs(scores_before[dim] - scores_after[dim])).item()
        print(f"  Dimension: {dim:<20} | Max diff: {diff:.6e}")
        assert diff < 1e-5, f"Checkpoints round-trip diff for {dim} exceeds 1e-5: {diff}"
        
    print("[ROUND-TRIP VERIFICATION] Saved and reloaded model scores are IDENTICAL (< 1e-5).")
    print("=" * 50 + "\n")
    
    # Clean up temp test directory
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Self-verify Bradley-Terry loss math direction
    test_loss_direction()
    
    # 2. Distributed Setup
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
    
    # 3. Load Model and Tokenizer
    if is_main_process:
        print(f"Loading base model from: {args.model_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Configure rank-specific device mapping for 4-bit DDP loading compatibility
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
        print("Please verify paths or check fallback rules to Qwen3-4B.")
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
    
    # 4. Verify Optimizer Parameter Groups (requires both LoRA and Rating Heads to be trainable)
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                lora_params.append((name, param))
            elif "rating_heads" in name:
                head_params.append((name, param))
                
    if is_main_process:
        print("\n" + "=" * 50)
        print("OPTIMIZER PARAMETER GROUPS VERIFICATION")
        print("=" * 50)
        print(f"Total Trainable LoRA Parameters: {len(lora_params)}")
        print(f"Total Trainable Scalar Head Parameters: {len(head_params)}")
        print("=" * 50 + "\n")
        
    # Enforce gradient constraints
    if args.use_lora:
        assert len(lora_params) > 0, "ERROR: No LoRA parameters are set as trainable!"
    assert len(head_params) > 0, "ERROR: Scalar rating head parameters are not trainable!"

    # 5. Run Checkpoint Round-Trip Verification Test
    if is_main_process:
        verify_checkpoint_round_trip(model, tokenizer, args, device)

    # 6. Load datasets
    train_dataset = PairwiseDataset(args.train_file, tokenizer, args.max_length, args.max_train_samples)
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
        val_dataset = PairwiseDataset(args.validation_file, tokenizer, args.max_length, args.max_eval_samples)
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

    # 7. Training Loop with Benchmark
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
            prob_labels = {k: v.to(device) for k, v in batch["prob_labels"].items()}
            
            # Forward pass
            f_start = time.time()
            ratings_a = model(input_ids_a, attention_mask_a)
            ratings_b = model(input_ids_b, attention_mask_b)
            
            batch_loss = 0.0
            for dim in QUALITY_DIMENSIONS:
                dim_loss = bradley_terry_loss(ratings_a[dim], ratings_b[dim], prob_labels[dim])
                batch_loss += dim_loss
            batch_loss = batch_loss / len(QUALITY_DIMENSIONS)
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
                
                # Check rating heads and LoRA parameters have non-zero gradients
                if step < 10 and is_main_process:
                    head_grad_norm = sum(p.grad.norm().item() for p in model.rating_heads.parameters() if p.grad is not None)
                    print(f"  [GRAD VERIFY] Rating Heads Gradient Norm: {head_grad_norm:.6f}")
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
            save_modular_checkpoint(model, tokenizer, checkpoint_dir, args)
            
    if is_main_process:
        final_dir = os.path.join(args.output_dir, "final_qurater")
        save_modular_checkpoint(model, tokenizer, final_dir, args)

if __name__ == "__main__":
    main()
