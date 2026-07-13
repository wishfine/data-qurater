#!/usr/bin/env python3
"""
Score corpus with trained QuRater model and select high-quality data
Implements the softmax sampling method from the QuRating paper
"""
import os
import json
import argparse
from typing import List, Dict, Optional

import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import jsonlines

from train_qurater import QuRaterModel, QUALITY_DIMENSIONS


def load_trained_model(model_dir: str, device: str = "cuda") -> tuple:
    """Load trained QuRater model from directory"""
    base_model_path = os.path.join(model_dir, "base_model")
    rating_heads_path = os.path.join(model_dir, "rating_heads.pt")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = QuRaterModel(base_model_path)
    model.rating_heads.load_state_dict(torch.load(rating_heads_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer


def score_texts(
    model: QuRaterModel,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512
) -> Dict[str, np.ndarray]:
    """Score a list of texts and return scores for each dimension"""
    all_scores = {dim: [] for dim in QUALITY_DIMENSIONS}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring corpus"):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            ratings = model(**encodings)
            for dim in QUALITY_DIMENSIONS:
                all_scores[dim].extend(ratings[dim].cpu().float().numpy())
    
    return {dim: np.array(scores) for dim, scores in all_scores.items()}


def softmax_sample(
    scores: np.ndarray,
    n_select: int,
    temperature: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    Softmax probability sampling without replacement as described in paper:
    P(d_i) = exp(s_i / τ) / Σ_j exp(s_j / τ)
    
    Optimized via Gumbel-Max Trick to reduce complexity from O(K * N) to O(N log N).
    It is mathematically equivalent to the sequential sampling but takes seconds instead of days.
    
    Args:
        scores: Array of scores for each document
        n_select: Number of documents to select
        temperature: τ parameter, controls diversity
            - τ → 0: equivalent to top-k selection (always pick highest scores)
            - τ → ∞: equivalent to uniform random sampling
        seed: Random seed
    
    Returns:
        Indices of selected documents
    """
    np.random.seed(seed)
    n = len(scores)
    
    if n_select >= n:
        # Select all elements if requested size is larger or equal
        return np.argsort(scores)[::-1]
        
    if temperature < 1e-6:
        # When temperature is near 0, it degenerates to deterministic Top-K selection
        print("Temperature is near 0, falling back to deterministic Top-K selection.")
        return np.argsort(scores)[-n_select:][::-1]
    
    # Draw independent standard Uniform samples
    u = np.random.uniform(low=1e-10, high=1.0, size=n)
    
    # Calculate Gumbel noise: G = -log(-log(U))
    gumbel_noise = -np.log(-np.log(u))
    
    # Calculate perturbed scores: s_i / tau + G_i
    perturbed_scores = scores / temperature + gumbel_noise
    
    # Efficiently partition to find the top n_select indices
    partitioned_indices = np.argpartition(perturbed_scores, -n_select)[-n_select:]
    
    # Sort the partitioned indices descending based on their perturbed scores
    sorted_partition_indices = np.argsort(perturbed_scores[partitioned_indices])[::-1]
    selected_indices = partitioned_indices[sorted_partition_indices]
    
    return selected_indices


def load_corpus(corpus_path: str, text_key: str = "text") -> List[str]:
    """Load corpus from jsonl file"""
    texts = []
    if corpus_path.endswith(".jsonl"):
        with jsonlines.open(corpus_path) as reader:
            for obj in reader:
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict) and text_key in obj:
                    texts.append(obj[text_key])
    elif corpus_path.endswith(".txt"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Score corpus and select high-quality data with QuRater")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with trained QuRater model")
    parser.add_argument("--input_corpus", type=str, required=True, help="Path to input corpus (jsonl/txt)")
    parser.add_argument("--output_file", type=str, required=True, help="Output jsonl file for selected data")
    parser.add_argument("--scores_output", type=str, default=None, help="Optional path to save all scores as json")
    parser.add_argument("--select_dimension", type=str, default="educational_value",
                        choices=QUALITY_DIMENSIONS + ["average"],
                        help="Which dimension to use for selection (paper finds educational_value works best)")
    parser.add_argument("--n_select", type=int, required=True, help="Number of documents to select")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Softmax temperature τ (lower = higher quality, lower diversity)")
    parser.add_argument("--batch_size", type=int, default=16, help="Scoring batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--text_key", type=str, default="text", help="Key for text field in jsonl input")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading trained QuRater model...")
    model, tokenizer = load_trained_model(args.model_dir, device)
    
    print(f"Loading corpus from {args.input_corpus}...")
    texts = load_corpus(args.input_corpus, args.text_key)
    print(f"Loaded {len(texts)} documents")
    
    if args.n_select > len(texts):
        print(f"Warning: n_select ({args.n_select}) > corpus size ({len(texts)}), selecting all documents")
        args.n_select = len(texts)
    
    print("Scoring all documents...")
    all_scores = score_texts(model, tokenizer, texts, device, args.batch_size, args.max_length)
    
    if args.scores_output:
        scores_to_save = {dim: scores.tolist() for dim, scores in all_scores.items()}
        with open(args.scores_output, "w", encoding="utf-8") as f:
            json.dump(scores_to_save, f)
        print(f"All scores saved to {args.scores_output}")
    
    # Get selection scores
    if args.select_dimension == "average":
        select_scores = np.mean([all_scores[dim] for dim in QUALITY_DIMENSIONS], axis=0)
        print("Using average score across all four dimensions for selection")
    else:
        select_scores = all_scores[args.select_dimension]
        print(f"Using {args.select_dimension} scores for selection")
    
    print(f"Selecting {args.n_select} documents with temperature τ={args.temperature}...")
    selected_indices = softmax_sample(select_scores, args.n_select, args.temperature, args.seed)
    
    print(f"Saving selected data to {args.output_file}...")
    with jsonlines.open(args.output_file, mode="w") as writer:
        for idx in tqdm(selected_indices, desc="Writing output"):
            writer.write({
                "text": texts[idx],
                "scores": {dim: float(all_scores[dim][idx]) for dim in QUALITY_DIMENSIONS}
            })
    
    # Print statistics
    selected_scores = select_scores[selected_indices]
    all_mean = np.mean(select_scores)
    selected_mean = np.mean(selected_scores)
    print(f"\nSelection complete!")
    print(f"Corpus score mean: {all_mean:.4f}")
    print(f"Selected score mean: {selected_mean:.4f} (improvement: {selected_mean - all_mean:.4f})")
    print(f"Selected {len(selected_indices)} documents out of {len(texts)} ({100*len(selected_indices)/len(texts):.1f}%)")


if __name__ == "__main__":
    main()
