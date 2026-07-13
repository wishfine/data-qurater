#!/usr/bin/env python3
"""
Generate pairwise comparison labels using GPT as described in the QuRating paper
"""
import os
import json
import time
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonlines
from tqdm import tqdm
from openai import OpenAI


DIMENSION_PROMPTS = {
    "writing_style": "Which text has a more polished and beautiful writing style? Consider language fluency, logical structure, and expressive quality.",
    "required_expertise": "Which text requires greater expertise and prerequisite knowledge to understand? Consider the depth and breadth of specialized domain knowledge needed.",
    "facts_trivia": "Which text contains more facts and trivia? Prefer specific facts and obscure trivia over more common knowledge, consider accuracy and richness of factual information.",
    "educational_value": "Which text has more educational value? Consider its effectiveness in imparting knowledge, developing skills, and inspiring thinking."
}

SYSTEM_PROMPT = """You are an expert text quality evaluator. You will be given two texts labeled Text A and Text B.
For each quality dimension, you need to output a probability value between 0 and 1 indicating how likely Text B is better than Text A on that dimension.
- Output 0.0 means Text A is definitely better
- Output 1.0 means Text B is definitely better  
- Output 0.5 means they are equal in quality
- Values between 0 and 1 indicate varying degrees of confidence

Only output a valid JSON object with four keys: writing_style, required_expertise, facts_trivia, educational_value.
Do not output any other explanation or text."""


def call_openai_for_pair(
    client: OpenAI,
    text_a: str,
    text_b: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    max_retries: int = 3
) -> Dict[str, float]:
    """Call GPT to get pairwise comparison probabilities"""
    user_prompt = f"""Text A:
{text_a}

Text B:
{text_b}

Compare these two texts on each quality dimension and output the probability that Text B is better than Text A."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content.strip())
            # Validate and clamp probabilities
            probs = {}
            for dim in DIMENSION_PROMPTS.keys():
                if dim in result:
                    val = float(result[dim])
                    probs[dim] = max(0.0, min(1.0, val))
                else:
                    probs[dim] = 0.5
            return probs
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed after {max_retries} retries: {e}")
                return {dim: 0.5 for dim in DIMENSION_PROMPTS.keys()}


def load_texts(text_file: str) -> List[str]:
    """Load texts from a file (one text per line or jsonl)"""
    texts = []
    if text_file.endswith(".jsonl"):
        with jsonlines.open(text_file) as reader:
            for obj in reader:
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict) and "text" in obj:
                    texts.append(obj["text"])
    elif text_file.endswith(".txt"):
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    elif text_file.endswith(".json"):
        with open(text_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
    return texts


def generate_pairs(texts: List[str], num_pairs: int, seed: int = 42) -> List[Tuple[int, int]]:
    """Generate random unique pairs of texts"""
    import random
    random.seed(seed)
    pairs = set()
    n = len(texts)
    while len(pairs) < num_pairs:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i != j:
            # Store as (min_idx, max_idx) to avoid duplicates but order doesn't matter since we compare A vs B
            pair = (i, j) if i < j else (j, i)
            pairs.add(pair)
    return list(pairs)


def main():
    parser = argparse.ArgumentParser(description="Generate pairwise comparison labels for QuRater training")
    parser.add_argument("--input_texts", type=str, required=True, help="Path to input texts file (txt/json/jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Output jsonl file for labeled pairs")
    parser.add_argument("--num_pairs", type=int, default=200000, help="Number of pairs to generate (paper uses 200k general + 50k domain)")
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel API calls")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_url", type=str, default=None, help="Optional OpenAI-compatible base URL")
    args = parser.parse_args()
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=args.base_url
    )
    
    print("Loading texts...")
    texts = load_texts(args.input_texts)
    print(f"Loaded {len(texts)} texts")
    
    print(f"Generating {args.num_pairs} random pairs...")
    pairs = generate_pairs(texts, args.num_pairs, args.seed)
    
    # Check existing output to resume
    processed_pairs = set()
    if os.path.exists(args.output_file):
        with jsonlines.open(args.output_file) as reader:
            for obj in reader:
                if isinstance(obj, dict) and "pair_idx" in obj:
                    processed_pairs.add(obj["pair_idx"])
        print(f"Resuming from {len(processed_pairs)} existing pairs")
    
    print(f"Starting API calls with {args.num_workers} workers...")
    results = []
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for pair_idx, (i, j) in enumerate(pairs):
            if pair_idx in processed_pairs:
                continue
            # Randomly decide order (A/B) to avoid position bias
            import random
            if random.random() > 0.5:
                i, j = j, i
            future = executor.submit(
                call_openai_for_pair,
                client,
                texts[i],
                texts[j],
                args.openai_model
            )
            futures[future] = (pair_idx, texts[i], texts[j])
        
        with jsonlines.open(args.output_file, mode="a") as writer:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating labels"):
                pair_idx, text_a, text_b = futures[future]
                probs = future.result()
                writer.write({
                    "pair_idx": pair_idx,
                    "text_a": text_a,
                    "text_b": text_b,
                    "probs": probs
                })
    
    print(f"Done! Labels saved to {args.output_file}")


if __name__ == "__main__":
    main()
