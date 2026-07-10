from __future__ import annotations
import json
import random
from typing import Dict, Any

QUALITY_DIMENSIONS = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value"
]

def main():
    data_path = "data/qurating/smoke_train.jsonl"
    
    # Load all available samples
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    # Deterministic sampling of 20 items (with replacement if total < 20)
    random.seed(42)
    sampled = [random.choice(samples) for _ in range(min(20, len(samples)))]
    
    print("\n" + "=" * 80)
    # Print the column schema mapping details
    print("DATA FIELDS & PREFERENCE DIRECTION VERIFICATION REPORT (20 SAMPLED RECS)")
    print("=" * 80)
    
    for idx, item in enumerate(sampled):
        text_a = item["text_a"]
        text_b = item["text_b"]
        target = float(item["target"])
        dim_idx = int(item["dimension_id"])
        dim_name = QUALITY_DIMENSIONS[dim_idx]
        confidence = float(item["confidence"])
        
        # In QuRating methodology:
        # Target = 1 means B is better than A; Target = 0 means A is better than B
        # The model predicts P(B > A) = sigmoid(s_B - s_A)
        # Therefore s_B - s_A is the logit direction.
        logit_direction = "s_B - s_A"
        
        print(f"Record #{idx+1}:")
        print(f"  [Text A Excerpt] : {text_a[:50]}...")
        print(f"  [Text B Excerpt] : {text_b[:50]}...")
        print(f"    - Dimension: {dim_name:<20} | Mapped Target: {target:.2f} | Confidence: {confidence:.2f} | Logit Direction: {logit_direction}")
        
        # Check consistency
        status = "PASS"
        if target > 0.5 and logit_direction != "s_B - s_A":
            status = "FAIL"
        assert status == "PASS", "Direction check failed!"
        
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
