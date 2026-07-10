from __future__ import annotations
import json
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any

QUALITY_DIMENSIONS = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value"
]

class PairExample:
    def __init__(self, text_a: str, text_b: str, probs: Dict[str, float]):
        self.text_a = text_a
        self.text_b = text_b
        self.probs = probs

class PairwiseDataset(Dataset):
    """
    Dataset for pairwise comparison text data with soft probability labels.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        max_samples: int | None = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path, max_samples)
        
    def _load_data(self, data_path: str, max_samples: int | None) -> List[PairExample]:
        examples = []
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        examples.append(self._parse_item(item))
        elif data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    examples.append(self._parse_item(item))
        else:
            raise ValueError(f"Unsupported data format: {data_path}, use .json or .jsonl")
            
        if max_samples is not None:
            examples = examples[:max_samples]
            
        # Print 3 samples for validation as requested by user constraints
        print(f"\n--- Loaded dataset: {data_path} (Total examples: {len(examples)}) ---")
        for idx in range(min(3, len(examples))):
            ex = examples[idx]
            print(f"Sample {idx+1}:")
            print(f"  [Text A]: {ex.text_a[:100]}...")
            print(f"  [Text B]: {ex.text_b[:100]}...")
            print(f"  [Probs] : {ex.probs}")
        print("---------------------------------------------------\n")
            
        return examples
        
    def _parse_item(self, item: Dict[str, Any]) -> PairExample:
        raw_probs = item.get("probs", {})
        # Map "facts_trivia" to "facts_and_trivia" to maintain compatibility
        probs = {
            "writing_style": float(raw_probs.get("writing_style", 0.5)),
            "required_expertise": float(raw_probs.get("required_expertise", 0.5)),
            "facts_and_trivia": float(raw_probs.get("facts_and_trivia", raw_probs.get("facts_trivia", 0.5))),
            "educational_value": float(raw_probs.get("educational_value", 0.5))
        }
        return PairExample(
            text_a=item["text_a"],
            text_b=item["text_b"],
            probs=probs
        )
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "text_a": ex.text_a,
            "text_b": ex.text_b,
            "probs": ex.probs
        }
        
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_a = [item["text_a"] for item in batch]
        texts_b = [item["text_b"] for item in batch]
        
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
