import os
import sys

# Add parent directory to sys.path to resolve root-level modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from transformers import AutoModel, AutoTokenizer

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from train_qurater_qwen import save_modular_checkpoint, save_experiment_metadata
from data.qurating_dataset import NormalizedPairwiseDataset

def save_baseline(model_path, val_file):
    print(f"=== INITIALIZING AND SAVING CHECKPOINT-0 BASELINE ===")
    print(f"Model Path: {model_path}")
    print(f"Val File:   {val_file}")
    
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    backbone = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # Wrap with PEFT LoRA configuration to match trainable model layout
    from peft import LoraConfig, get_peft_model
    target_modules = []
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name.split(".")[-1])
    target_modules = list(set(target_modules))
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    backbone = get_peft_model(backbone, peft_config)
    
    model = QwenQuRater(backbone=backbone)
    model.to(device)
    
    # Save validation manifest and experiment config
    val_dataset = None
    if val_file and os.path.exists(val_file):
        val_dataset = NormalizedPairwiseDataset(val_file, tokenizer, max_length=256)
        
    class DummyArgs:
        def __init__(self):
            self.model_path = model_path
            self.use_lora = True
            self.use_4bit = False
            self.bf16 = True
            self.seed = 42
            self.max_length = 256
            self.confidence_threshold = 0.5
            self.validation_file = val_file
            
    args = DummyArgs()
    
    # Save experiment configs
    save_experiment_metadata(args, val_dataset)
    
    # Save checkpoint-0
    checkpoint_0_dir = "outputs/qwen35_4b_experiment/checkpoint-0"
    
    save_modular_checkpoint(model, tokenizer, checkpoint_0_dir, args, epoch=0, target_modules=target_modules)
    print("Baseline saved successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default="")
    args = parser.parse_args()
    
    save_baseline(args.model_path, args.validation_file)
