import os
import sys
import json
import gc
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from train_qurater_qwen import save_modular_checkpoint

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Base model directory path")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for round-trip error")
    args = parser.parse_args()

    # 1. Fixed Seed
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing checkpoint round-trip on device: {device}")

    # 2. Load Qwen3-0.6B Base Model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    backbone = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    # 3. Mount LoRA
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
        lora_dropout=0.0,  # Set dropout to 0.0 to ensure deterministic round-trip
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    backbone = get_peft_model(backbone, lora_config)

    # 4. Initialize Rating Head
    model = QwenQuRater(backbone=backbone)
    model.to(device)

    # 5. Tokenized Fixed Inputs
    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long).to(device)

    # 6. Execute 1 Optimizer Step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    scores = model(input_ids, attention_mask)
    loss = scores.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 7. Model Eval Mode
    model.eval()

    # 8. Calculate and Record Score Before Save
    with torch.no_grad():
        score_before_tensor = model(input_ids, attention_mask)
        score_before = score_before_tensor.float().cpu().numpy().tolist()

    # 9. Save Modular Checkpoint
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

    # 10. Delete Model Object
    del model
    del backbone
    del optimizer
    
    # 11. Run GC
    gc.collect()
    
    # 12. Empty CUDA Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 13-15. Reload from Checkpoint
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
    
    # Load safe rating head
    heads_path = os.path.join(checkpoint_dir, "rating_head.safetensors")
    if os.path.exists(heads_path):
        from safetensors.torch import load_file
        model_new.score.load_state_dict(load_file(heads_path, map_location=device))
    else:
        pt_path = os.path.join(checkpoint_dir, "rating_head.pt")
        model_new.score.load_state_dict(torch.load(pt_path, map_location=device))
        
    model_new.to(device)

    # 16. Model Eval Mode
    model_new.eval()

    # 17. Calculate Score After Reload
    with torch.no_grad():
        score_after_tensor = model_new(input_ids, attention_mask)
        score_after = score_after_tensor.float().cpu().numpy().tolist()

    # 18. Calculate max_abs_diff
    max_abs_diff = float(np.max(np.abs(np.array(score_before) - np.array(score_after))))
    print(f"Score Before Reload: {score_before}")
    print(f"Score After Reload:  {score_after}")
    print(f"Max Absolute Diff:   {max_abs_diff:.6e}")

    # 19. Verify status against tolerance
    status = "PASS" if max_abs_diff <= args.tolerance else "FAIL"
    print(f"Verification Status: {status}")

    # Save outputs to reports/server/checkpoint_roundtrip.json
    os.makedirs("reports/server", exist_ok=True)
    report = {
        "score_before": score_before,
        "score_after": score_after,
        "max_abs_diff": max_abs_diff,
        "tolerance": args.tolerance,
        "status": status
    }
    with open("reports/server/checkpoint_roundtrip.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Cleanup test directory
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    if status == "FAIL":
        sys.exit(1)

if __name__ == "__main__":
    main()
