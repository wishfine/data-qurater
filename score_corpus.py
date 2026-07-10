from __future__ import annotations
import os
import sys
import json
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Any

from models.qwen_qurater import QwenQuRater, QUALITY_DIMENSIONS

def chunk_text(text: str, tokenizer, max_tokens: int = 512) -> List[Dict[str, Any]]:
    """Split text into non-overlapping token windows of up to max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return [{"text": "", "length": 1}]
        
    if len(tokens) <= max_tokens:
        return [{"text": text, "length": len(tokens)}]
        
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append({
            "text": chunk_text,
            "length": len(chunk_tokens)
        })
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Score text corpus with QwenQuRater")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model directory")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to modular checkpoint directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw corpus JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output scored JSONL file")
    parser.add_argument("--max_length", type=int, default=512, help="Chunk token window limit")
    parser.add_argument("--pooling_type", type=str, default="last_token", help="Pooling strategy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model & Tokenizer
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
    
    # Load LoRA adapter if use_lora is true in config metadata
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

    # 2. Iterate and score documents
    print(f"Scoring documents from: {args.input_file}")
    
    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file does not exist: {args.input_file}")
        sys.exit(1)
        
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:
             
        for line in tqdm(fin, desc="Scoring corpus"):
            if not line.strip():
                continue
            item = json.loads(line)
            
            doc_text = item.get("text", item.get("content", ""))
            
            # Segment documents into token chunks (stride=max_length, overlap=0, add_special_tokens=False)
            # matching the official tokenize_and_chunk logic
            chunks = chunk_text(doc_text, tokenizer, args.max_length)
            
            chunk_texts = [c["text"] for c in chunks]
            
            with torch.no_grad():
                encodings = tokenizer(
                    chunk_texts,
                    truncation=True,
                    max_length=args.max_length,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                
                ratings = model(encodings["input_ids"], encodings["attention_mask"])
                
            chunk_ratings = []
            for i in range(len(chunks)):
                ratings_i = {dim: float(ratings[dim][i].cpu().float().item()) for dim in QUALITY_DIMENSIONS}
                chunk_ratings.append(ratings_i)
                
            # Length-weighted aggregation of scores
            weighted_scores = {dim: 0.0 for dim in QUALITY_DIMENSIONS}
            total_weight = sum(c["length"] for c in chunks)
            
            for chunk, scores in zip(chunks, chunk_ratings):
                weight = chunk["length"]
                for dim in QUALITY_DIMENSIONS:
                    weighted_scores[dim] += scores[dim] * weight
                    
            for dim in QUALITY_DIMENSIONS:
                weighted_scores[dim] = weighted_scores[dim] / total_weight
                
            item["qurating_scores"] = weighted_scores
            item["qurating_chunks"] = chunk_ratings
            
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Scoring complete. Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()
