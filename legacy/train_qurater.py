#!/usr/bin/env python3
"""
QuRater: Quality Rating Model Training Script
Based on paper: QuRating: Selecting High-Quality Data for Training Language Models (ICML 2024)
"""
import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np


QUALITY_DIMENSIONS = [
    "writing_style",
    "required_expertise", 
    "facts_trivia",
    "educational_value"
]


@dataclass
class PairExample:
    text_a: str
    text_b: str
    # Probability that B > A for each dimension, value in [0.0, 1.0]
    probs: Dict[str, float]


class QuRaterModel(nn.Module):
    """
    QuRater Model: Base transformer + 4 linear heads for 4 quality dimensions
    Paper: Fine-tuned Sheared-Llama-1.3B with 4 linear heads
    """
    def __init__(self, model_name_or_path: str, hidden_size: Optional[int] = None):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        if hidden_size is None:
            hidden_size = self.base_model.config.hidden_size
        
        # Four independent rating heads for each quality dimension
        self.rating_heads = nn.ModuleDict({
            dim: nn.Linear(hidden_size, 1, bias=False) 
            for dim in QUALITY_DIMENSIONS
        })
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use last token hidden state as sentence representation (as in reward modeling)
        last_hidden = outputs.last_hidden_state
        # Mean pooling over non-padding tokens for better representation
        hidden_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = torch.sum(last_hidden * hidden_mask, 1) / torch.clamp(hidden_mask.sum(1), min=1e-9)
        
        # Compute scalar rating for each dimension
        ratings = {}
        for dim in QUALITY_DIMENSIONS:
            ratings[dim] = self.rating_heads[dim](pooled).squeeze(-1)
        return ratings
    
    def get_scores(self, texts: List[str], tokenizer, device: str = "cuda", max_length: int = 512) -> Dict[str, List[float]]:
        """Get quality scores for a list of individual texts"""
        self.eval()
        all_scores = {dim: [] for dim in QUALITY_DIMENSIONS}
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), 8), desc="Scoring texts"):
                batch_texts = texts[i:i+8]
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                
                ratings = self(**encodings)
                for dim in QUALITY_DIMENSIONS:
                    all_scores[dim].extend(ratings[dim].cpu().float().tolist())
        
        return all_scores


class PairwiseDataset(Dataset):
    """Dataset for pairwise comparison data with soft probability labels"""
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[PairExample]:
        examples = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        examples.append(PairExample(
                            text_a=item["text_a"],
                            text_b=item["text_b"],
                            probs={k: float(v) for k, v in item["probs"].items()}
                        ))
        elif data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    examples.append(PairExample(
                        text_a=item["text_a"],
                        text_b=item["text_b"],
                        probs={k: float(v) for k, v in item["probs"].items()}
                    ))
        else:
            raise ValueError(f"Unsupported data format: {data_path}, use .json or .jsonl")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        return {
            "text_a": ex.text_a,
            "text_b": ex.text_b,
            "probs": ex.probs
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        texts_a = [item["text_a"] for item in batch]
        texts_b = [item["text_b"] for item in batch]
        batch_size = len(batch)
        
        encodings_a = self.tokenizer(
            texts_a,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        encodings_b = self.tokenizer(
            texts_b,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        # Stack probability labels
        prob_labels = {}
        for dim in QUALITY_DIMENSIONS:
            prob_labels[dim] = torch.tensor(
                [item["probs"].get(dim, 0.5) for item in batch],
                dtype=torch.float32
            )
        
        return {
            "input_ids_a": encodings_a["input_ids"],
            "attention_mask_a": encodings_a["attention_mask"],
            "input_ids_b": encodings_b["input_ids"],
            "attention_mask_b": encodings_b["attention_mask"],
            "prob_labels": prob_labels
        }


def bradley_terry_loss(
    ratings_a: torch.Tensor,
    ratings_b: torch.Tensor,
    p_b_gt_a: torch.Tensor
) -> torch.Tensor:
    """
    Bradley-Terry Model Loss for soft labels
    P(B > A) = sigmoid(rating_b - rating_a)
    Loss = Binary Cross Entropy between predicted probability and ground truth probability
    """
    # Cast tensors to float32 for numerical stability and compatibility with mixed-precision training
    r_a = ratings_a.float()
    r_b = ratings_b.float()
    p_gt = p_b_gt_a.float()
    
    logits = r_b - r_a
    # Clamp for numerical stability
    p_b_pred = torch.sigmoid(logits)
    p_b_pred = torch.clamp(p_b_pred, min=1e-7, max=1-1e-7)
    
    # Binary cross entropy with soft labels
    loss = -(p_gt * torch.log(p_b_pred) + (1 - p_gt) * torch.log(1 - p_b_pred))
    return loss.mean()


def train_epoch(
    model: QuRaterModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    model.train()
    total_losses = {dim: 0.0 for dim in QUALITY_DIMENSIONS}
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        prob_labels = {k: v.to(device) for k, v in batch["prob_labels"].items()}
        
        # Forward pass for both texts
        ratings_a = model(input_ids_a, attention_mask_a)
        ratings_b = model(input_ids_b, attention_mask_b)
        
        # Compute loss for each dimension
        dim_losses = {}
        batch_loss = 0.0
        for dim in QUALITY_DIMENSIONS:
            dim_loss = bradley_terry_loss(ratings_a[dim], ratings_b[dim], prob_labels[dim])
            dim_losses[dim] = dim_loss
            batch_loss = batch_loss + dim_loss
        
        batch_loss = batch_loss / len(QUALITY_DIMENSIONS)
        batch_loss = batch_loss / grad_accum_steps
        batch_loss.backward()
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update metrics
        for dim in QUALITY_DIMENSIONS:
            total_losses[dim] += dim_losses[dim].item()
        total_loss += batch_loss.item() * grad_accum_steps
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{total_loss/num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    avg_losses = {dim: total_losses[dim]/num_batches for dim in QUALITY_DIMENSIONS}
    avg_losses["total"] = total_loss / num_batches
    return avg_losses


def evaluate(
    model: QuRaterModel,
    dataloader: DataLoader,
    device: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    total_losses = {dim: 0.0 for dim in QUALITY_DIMENSIONS}
    correct = {dim: 0 for dim in QUALITY_DIMENSIONS}
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            prob_labels = {k: v.to(device) for k, v in batch["prob_labels"].items()}
            
            ratings_a = model(input_ids_a, attention_mask_a)
            ratings_b = model(input_ids_b, attention_mask_b)
            
            batch_size = input_ids_a.size(0)
            total += batch_size
            
            for dim in QUALITY_DIMENSIONS:
                dim_loss = bradley_terry_loss(ratings_a[dim], ratings_b[dim], prob_labels[dim])
                total_losses[dim] += dim_loss.item() * batch_size
                # Accuracy: predict B better when predicted p > 0.5, match ground truth p > 0.5
                pred = (ratings_b[dim] > ratings_a[dim]).long()
                gt = (prob_labels[dim] > 0.5).long()
                correct[dim] += (pred == gt).sum().item()
    
    avg_losses = {dim: total_losses[dim]/total for dim in QUALITY_DIMENSIONS}
    accuracies = {dim: correct[dim]/total for dim in QUALITY_DIMENSIONS}
    return avg_losses, accuracies


def main():
    parser = argparse.ArgumentParser(description="Train QuRater quality rating model")
    parser.add_argument("--model_name_or_path", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B",
                        help="Base model name or path (paper uses Sheared-LLaMA-1.3B)")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (json/jsonl)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data (optional)")
    parser.add_argument("--output_dir", type=str, default="./qurater_model", help="Output directory for saved model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = QuRaterModel(args.model_name_or_path)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}, Trainable parameters: {trainable_params:,}")
    
    print("Loading datasets...")
    train_dataset = PairwiseDataset(args.train_data, tokenizer, args.max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0
    )
    
    val_dataloader = None
    if args.val_data:
        val_dataset = PairwiseDataset(args.val_data, tokenizer, args.max_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=0
        )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_dataloader) * args.num_epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Starting training for {args.num_epochs} epochs...")
    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{args.num_epochs} ===")
        train_losses = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            device, args.grad_accum_steps, args.max_grad_norm
        )
        
        print(f"Train losses: total={train_losses['total']:.4f}")
        for dim in QUALITY_DIMENSIONS:
            print(f"  {dim}: {train_losses[dim]:.4f}")
        
        if val_dataloader:
            val_losses, val_accs = evaluate(model, val_dataloader, device)
            avg_val_acc = sum(val_accs.values()) / len(QUALITY_DIMENSIONS)
            print(f"\nValidation results:")
            for dim in QUALITY_DIMENSIONS:
                print(f"  {dim}: loss={val_losses[dim]:.4f}, acc={val_accs[dim]:.4f}")
            print(f"  Average accuracy: {avg_val_acc:.4f}")
            
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                print(f"New best model found! Saving to {args.output_dir}")
                model.base_model.save_pretrained(os.path.join(args.output_dir, "base_model"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "base_model"))
                torch.save(model.rating_heads.state_dict(), os.path.join(args.output_dir, "rating_heads.pt"))
                with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                    json.dump(vars(args), f, indent=2)
    
    if not val_dataloader:
        print(f"Saving final model to {args.output_dir}")
        model.base_model.save_pretrained(os.path.join(args.output_dir, "base_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "base_model"))
        torch.save(model.rating_heads.state_dict(), os.path.join(args.output_dir, "rating_heads.pt"))
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    print("\nTraining complete!")
    
    if val_dataloader:
        print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
