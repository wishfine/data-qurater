from __future__ import annotations
import json
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any
from qurater_utils import validate_normalized_record

DIMENSION_NAMES = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value",
]

class NormalizedPairwiseDataset(Dataset):
    """
    Unified Pairwise Dataset.
    Loads data in the internal normalized format:
    {
      "text_a": "...",
      "text_b": "...",
      "target": 0.8,
      "dimension_id": 0,
      "confidence": 0.6,
      "domain": "optional"
    }
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
        self.tokenizer.padding_side = "right"  # Force padding side to right
        self.data_path = data_path
        self.file_handle = None
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Detect format of first valid line
        self.is_flat_format = False
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if "probs" in item and "target" not in item:
                            self.is_flat_format = True
                    except Exception:
                        pass
                    break
                    
        # Force in-memory loading for small subsets (<= 200000 samples) for speed and DDP multiprocessing safety
        if max_samples is not None and max_samples <= 200000:
            print(f"[INFO] Loading small dataset {data_path} in-memory ({max_samples} samples max)...")
            self.use_lazy_loading = False
            self.examples = self._load_data(data_path, max_samples)
        elif self.is_flat_format:
            print(f"[INFO] File {data_path} is in legacy flat format. Loading in-memory...")
            self.use_lazy_loading = False
            self.examples = self._load_data(data_path, max_samples)
        else:
            print(f"[INFO] High-performance indexing dataset {data_path} (Lazy Loading)...")
            self.use_lazy_loading = True
            
            # Read in 1MB chunks to minimize NFS latency and bypass garbage collector overhead
            self.offsets = [0]
            with open(data_path, "rb") as f:
                chunk_size = 1024 * 1024  # 1MB buffer
                offset = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    start = 0
                    while True:
                        pos = chunk.find(b"\n", start)
                        if pos == -1:
                            break
                        self.offsets.append(offset + pos + 1)
                        start = pos + 1
                    
                    offset += len(chunk)
            
            # Remove trailing EOF offset if present
            if self.offsets and self.offsets[-1] >= offset:
                self.offsets.pop()
                
            if max_samples is not None:
                self.offsets = self.offsets[:max_samples]
                
            print(f"\n--- Loaded Normalized Dataset (Indexed): {data_path} (Total samples: {len(self.offsets)}) ---")
            for idx in range(min(3, len(self.offsets))):
                ex = self[idx]
                print(f"Sample {idx+1}:")
                print(f"  [Text A]: {ex['text_a'][:80]}...")
                print(f"  [Text B]: {ex['text_b'][:80]}...")
                print(f"  [Target] : {ex['target']:.4f} | Dim ID: {ex['dimension_id']} ({DIMENSION_NAMES[ex['dimension_id']]}) | Confidence: {ex['confidence']:.4f}")
            print("---------------------------------------------------\n")
            
            # Close file handle in parent process to avoid sharing descriptor across forks in DataLoader workers
            if self.file_handle is not None:
                self.file_handle.close()
                self.file_handle = None
        
    def _load_data(self, data_path: str, max_samples: int | None) -> List[Dict[str, Any]]:
        examples = []
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                
                # Check if it is already in the normalized format
                if "target" in item and "dimension_id" in item:
                    validate_normalized_record(item)
                    examples.append(item)
                # Check if it is in the raw Flat Pairwise format (from score_pairwise.py or raw JSONL)
                elif "text_a" in item and "text_b" in item and "probs" in item:
                    # Expand one raw pair into 4 normalized records (one per quality dimension)
                    raw_probs = item["probs"]
                    for dim_idx, dim_name in enumerate(DIMENSION_NAMES):
                        # Map raw "facts_trivia" to "facts_and_trivia"
                        raw_key = "facts_trivia" if dim_name == "facts_and_trivia" else dim_name
                        target = float(raw_probs.get(dim_name, raw_probs.get(raw_key, 0.5)))
                        confidence = 2.0 * abs(target - 0.5)
                        
                        examples.append({
                            "text_a": item["text_a"],
                            "text_b": item["text_b"],
                            "target": target,
                            "dimension_id": dim_idx,
                            "confidence": confidence,
                            "domain": item.get("domain", "general")
                        })
                else:
                    raise ValueError(f"Unrecognized data format in {data_path}")
                    
        if max_samples is not None:
            examples = examples[:max_samples]
            
        print(f"\n--- Loaded Normalized Dataset: {data_path} (Total pairwise samples: {len(examples)}) ---")
        for idx in range(min(3, len(examples))):
            ex = examples[idx]
            print(f"Sample {idx+1}:")
            print(f"  [Text A]: {ex['text_a'][:80]}...")
            print(f"  [Text B]: {ex['text_b'][:80]}...")
            print(f"  [Target] : {ex['target']:.4f} | Dim ID: {ex['dimension_id']} ({DIMENSION_NAMES[ex['dimension_id']]}) | Confidence: {ex['confidence']:.4f}")
        print("---------------------------------------------------\n")
        
        return examples
        
    def __len__(self) -> int:
        if self.use_lazy_loading:
            return len(self.offsets)
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.use_lazy_loading:
            if self.file_handle is None:
                self.file_handle = open(self.data_path, "r", encoding="utf-8")
            self.file_handle.seek(self.offsets[idx])
            line = self.file_handle.readline()
            return json.loads(line)
        return self.examples[idx]
        
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        for item in batch:
            validate_normalized_record(item)
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
        
        targets = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
        dimension_ids = torch.tensor([item["dimension_id"] for item in batch], dtype=torch.long)
        confidences = torch.tensor([item["confidence"] for item in batch], dtype=torch.float32)
        
        return {
            "input_ids_a": encodings_a["input_ids"],
            "attention_mask_a": encodings_a["attention_mask"],
            "input_ids_b": encodings_b["input_ids"],
            "attention_mask_b": encodings_b["attention_mask"],
            "targets": targets,
            "dimension_ids": dimension_ids,
            "confidences": confidences,
            "domains": [item.get("domain", "general") for item in batch]
        }

class OfficialQuRatingDatasetAdapter:
    """
    Adapter to convert official QuRating matrix-based dataset files
    (having 'texts' lists and 'calibrated_predictions' matrices)
    into the internal normalized pairwise jsonl format.
    """
    @staticmethod
    def convert_file(input_path: str, output_path: str):
        print(f"Converting official dataset from {input_path} to {output_path} ...")
        import ast
        
        # Load raw lines (expecting jsonl or json)
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    
        total_converted = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for item in records:
                texts = item.get("texts", [])
                if isinstance(texts, str):
                    try:
                        texts = ast.literal_eval(texts)
                    except Exception:
                        continue
                        
                K = len(texts)
                if K < 2:
                    continue
                    
                # Determine domain(s)
                domain = item.get("domain", item.get("source_domains", "general"))
                if isinstance(domain, str):
                    if domain.startswith("[") and domain.endswith("]"):
                        try:
                            domain = ast.literal_eval(domain)
                        except Exception:
                            pass
                
                # Check format types
                # Format A (Old): 'calibrated_predictions' is a 3D matrix (K, K, 4)
                # Format B (New): separate 'writing_style_average', etc. 2D matrices (K, K)
                calibrated = item.get("calibrated_predictions", [])
                if isinstance(calibrated, str):
                    try:
                        calibrated = ast.literal_eval(calibrated)
                    except Exception:
                        calibrated = []
                
                has_calibrated = (isinstance(calibrated, list) and len(calibrated) > 0)
                
                matrices = {}
                for dim_idx, dim_name in enumerate(DIMENSION_NAMES):
                    key = f"{dim_name}_average"
                    raw_mat = item.get(key)
                    if raw_mat is not None:
                        if isinstance(raw_mat, str):
                            try:
                                matrices[dim_idx] = ast.literal_eval(raw_mat)
                            except Exception:
                                pass
                        elif isinstance(raw_mat, list):
                            matrices[dim_idx] = raw_mat
                            
                has_matrices = (len(matrices) == len(DIMENSION_NAMES))
                
                if not has_calibrated and not has_matrices:
                    continue
                    
                for i in range(K):
                    for j in range(K):
                        if i == j:
                            continue
                            
                        # Map domains
                        if isinstance(domain, list):
                            try:
                                pair_domain = f"{domain[i]}_{domain[j]}"
                            except IndexError:
                                pair_domain = "general"
                        else:
                            pair_domain = str(domain)
                            
                        for dim_idx in range(len(DIMENSION_NAMES)):
                            target = None
                            if has_calibrated:
                                try:
                                    target = float(calibrated[i][j][dim_idx])
                                except (IndexError, TypeError):
                                    pass
                            elif dim_idx in matrices:
                                try:
                                    target = float(matrices[dim_idx][i][j])
                                except (IndexError, TypeError):
                                    pass
                                    
                            if target is None or target == -100:
                                continue
                                
                            confidence = 2.0 * abs(target - 0.5)
                            
                            pairwise_record = {
                                "text_a": texts[i],
                                "text_b": texts[j],
                                "target": target,
                                "dimension_id": dim_idx,
                                "confidence": confidence,
                                "domain": pair_domain
                            }
                            fout.write(json.dumps(pairwise_record, ensure_ascii=False) + "\n")
                            total_converted += 1
                            
        print(f"Finished conversion. Generated {total_converted} pairwise records.")
