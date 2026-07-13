from __future__ import annotations

import os
from typing import Any, Dict


def resolve_run_paths(output_dir: str) -> Dict[str, str]:
    output_dir = os.path.normpath(output_dir)
    experiment_dir = os.path.dirname(output_dir)
    return {
        "experiment_dir": experiment_dir,
        "checkpoint_0": os.path.join(experiment_dir, "checkpoint-0"),
        "metadata_dir": experiment_dir,
        "evaluations_dir": os.path.join(experiment_dir, "evaluations"),
    }


def validate_normalized_record(record: Dict[str, Any]) -> None:
    required = ("text_a", "text_b", "target", "dimension_id", "confidence")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Normalized record is missing fields: {missing}")
    if not isinstance(record["text_a"], str) or not isinstance(record["text_b"], str):
        raise ValueError("text_a and text_b must be strings")
    target = float(record["target"])
    confidence = float(record["confidence"])
    dimension_id = int(record["dimension_id"])
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target must be within [0, 1], got {target}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be within [0, 1], got {confidence}")
    if dimension_id not in range(4):
        raise ValueError(f"dimension_id must be in [0, 3], got {dimension_id}")
