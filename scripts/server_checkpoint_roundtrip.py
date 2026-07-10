import os
import sys

# Add parent directory to sys.path to resolve root-level modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gc
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from train_qurater_qwen import save_modular_checkpoint, bradley_terry_loss
from data.qurating_dataset import NormalizedPairwiseDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Base model directory path")
    parser.add_argument("--eval_file", type=str, required=True, help="Data file to initialize dataset collator")
    parser.add_argument("--roundtrip_tolerance", type=float, default=1e-3, help="Tolerance for round-trip error")
    args = parser.parse_args()

    # 1. Fixed Seed
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing checkpoint round-trip on device: {device}")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Qwen3.5-4B Base Model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    backbone = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    # 4. Mount LoRA adapter
    from peft import LoraConfig, get_peft_model
    target_modules = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear):
            target_modules.append(name.split(".")[-1])
    target_modules = list(set(target_modules))
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,  # Zero dropout to prevent stochastic outputs during evaluation
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    backbone = get_peft_model(backbone, lora_config)

    # 5. Initialize Rating Head
    model = QwenQuRater(backbone=backbone)
    model.to(device)

    # 6. Construct standard training batch via collate_fn
    mock_raw_batch = [
        {
            "text_a": "This is a quality control test paragraph for A.",
            "text_b": "This represents a highly educational segment explaining complex astrophysics.",
            "target": 0.85,
            "dimension_id": 3,  # educational_value
            "confidence": 0.70,
            "domain": "science"
        },
        {
            "text_a": "Let us check vocabulary and syntactic fluency here.",
            "text_b": "Check syntax and sentence structures.",
            "target": 0.20,
            "dimension_id": 0,  # writing_style
            "confidence": 0.60,
            "domain": "general"
        }
    ]
    
    # Load dataset structure
    dataset = NormalizedPairwiseDataset(args.eval_file, tokenizer, max_length=256)
    batch = dataset.collate_fn(mock_raw_batch)

    # 7. Collect inputs
    input_ids_a = batch["input_ids_a"].to(device)
    attention_mask_a = batch["attention_mask_a"].to(device)
    input_ids_b = batch["input_ids_b"].to(device)
    attention_mask_b = batch["attention_mask_b"].to(device)
    targets = batch["targets"].to(device)
    dimension_ids = batch["dimension_ids"].to(device)
    confidences = batch["confidences"].to(device)

    # 8. Train mode forward & backward
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    ratings_a = model(input_ids_a, attention_mask_a, dimension_ids)
    ratings_b = model(input_ids_b, attention_mask_b, dimension_ids)
    
    loss = bradley_terry_loss(ratings_a, ratings_b, targets, confidences, 0.0)
    loss.backward()

    # Get gradient norms before optimizer step
    rating_head_grad_norm = sum(p.grad.norm().item() for p in model.score.parameters() if p.grad is not None)
    lora_grad_norm = sum(p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None and "lora_" in name)

    # Hard assertions of active gradients
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert rating_head_grad_norm > 0, "Rating head score weights have zero gradients!"
    assert lora_grad_norm > 0, "LoRA weights have zero gradients!"

    # 9. Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # 10. Eval mode
    model.eval()

    # 11. Record score before save (using base outputs shape (batch, 4))
    with torch.no_grad():
        score_before_tensor = model(input_ids_a, attention_mask_a)
        score_before = score_before_tensor.float().cpu().numpy().tolist()

    # 12. Save modular checkpoint
    checkpoint_dir = "./outputs/roundtrip_test"
    
    class DummyArgs:
        def __init__(self, model_path):
            self.model_path = model_path
            self.use_lora = True
            self.use_4bit = False
            self.bf16 = torch.cuda.is_available()
            self.seed = 42
            self.max_length = 256
            
    dummy_args = DummyArgs(args.model_path)
    save_modular_checkpoint(model, tokenizer, checkpoint_dir, dummy_args, epoch=1, target_modules=target_modules)

    # 13. Delete model object and clean memory
    del model
    del backbone
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 14. Reload from saved modular checkpoint
    print("Re-loading from saved modular checkpoint...")
    backbone_new = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    model_new = QwenQuRater(backbone=backbone_new)
    
    from peft import PeftModel
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    model_new.backbone = PeftModel.from_pretrained(model_new.backbone, adapter_dir)
    
    heads_path = os.path.join(checkpoint_dir, "rating_head.safetensors")
    if os.path.exists(heads_path):
        from safetensors.torch import load_file
        model_new.score.load_state_dict(load_file(heads_path, map_location=device))
    else:
        pt_path = os.path.join(checkpoint_dir, "rating_head.pt")
        model_new.score.load_state_dict(torch.load(pt_path, map_location=device))
        
    model_new.to(device)
    model_new.eval()

    # 15. Calculate score after reload
    with torch.no_grad():
        score_after_tensor = model_new(input_ids_a, attention_mask_a)
        score_after = score_after_tensor.float().cpu().numpy().tolist()

    # 16. Calculate metrics
    abs_diffs = np.abs(np.array(score_before) - np.array(score_after))
    max_abs_diff = float(np.max(abs_diffs))
    mean_abs_diff = float(np.mean(abs_diffs))
    
    print(f"Max Absolute Diff: {max_abs_diff:.6e} | Mean Absolute Diff: {mean_abs_diff:.6e}")
    status = "PASS" if max_abs_diff <= args.roundtrip_tolerance else "FAIL"
    print(f"Verification Status: {status}")

    # Output to reports/server/checkpoint_roundtrip.json
    os.makedirs("reports/server", exist_ok=True)
    report = {
        "loss": float(loss.item()),
        "rating_head_grad_norm": float(rating_head_grad_norm),
        "lora_grad_norm": float(lora_grad_norm),
        "score_before": score_before,
        "score_after": score_after,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "tolerance": args.roundtrip_tolerance,
        "status": status
    }
    with open("reports/server/checkpoint_roundtrip.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Cleanup
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    if status == "FAIL":
        sys.exit(1)

if __name__ == "__main__":
    main()
