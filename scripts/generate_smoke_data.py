"""Create a deterministic, text-disjoint QuRating smoke dataset.

The generated source file uses the raw pairwise ``probs`` schema.  The same
connected-component splitter used on the server then creates normalized train
and evaluation files, so the checked-in smoke data also passes leakage audits.
"""
from __future__ import annotations

import json
import os
import random

from build_smoke_split import build_split


def generate_smoke_data(output_dir: str = "data/qurating", seed: int = 42) -> None:
    random_generator = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    source_path = os.path.join(output_dir, "smoke_train_source.jsonl")
    train_path = os.path.join(output_dir, "smoke_train.jsonl")
    eval_path = os.path.join(output_dir, "smoke_eval.jsonl")
    manifest_path = os.path.join(output_dir, "smoke_split_manifest.json")

    records = []
    for index in range(12):
        text_a = (
            f"Smoke example {index} is a fragmented note with little explanation "
            "and no supporting detail."
        )
        text_b = (
            f"Smoke example {index} explains a scientific concept with clear structure, "
            "specific factual detail, and an educational takeaway."
        )
        records.append({
            "text_a": text_a,
            "text_b": text_b,
            "probs": {
                "writing_style": round(random_generator.uniform(0.75, 0.95), 4),
                "required_expertise": round(random_generator.uniform(0.55, 0.8), 4),
                "facts_trivia": round(random_generator.uniform(0.65, 0.9), 4),
                "educational_value": round(random_generator.uniform(0.75, 0.95), 4),
            },
            "domain": "synthetic_smoke",
        })

    with open(source_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    build_split(
        source_path,
        train_path,
        eval_path,
        manifest_path,
        raw_train_target=4,
        raw_eval_target=4,
        seed=seed,
    )


if __name__ == "__main__":
    generate_smoke_data()
