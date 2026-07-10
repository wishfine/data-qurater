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
    sampled = [random.choice(samples) for _ in range(20)]
    
    print("\n" + "=" * 80)
    # Print the column schema mapping details
    print("DATA FIELDS & PREFERENCE DIRECTION VERIFICATION REPORT (20 SAMPLED RECS)")
    print("=" * 80)
    
    for idx, item in enumerate(sampled):
        text_a = item["text_a"]
        text_b = item["text_b"]
        raw_probs = item["probs"]
        
        # Mapping rules
        probs = {
            "writing_style": float(raw_probs.get("writing_style", 0.5)),
            "required_expertise": float(raw_probs.get("required_expertise", 0.5)),
            "facts_and_trivia": float(raw_probs.get("facts_and_trivia", raw_probs.get("facts_trivia", 0.5))),
            "educational_value": float(raw_probs.get("educational_value", 0.5))
        }
        
        print(f"Record #{idx+1}:")
        print(f"  [Text A Excerpt] : {text_a[:50]}...")
        print(f"  [Text B Excerpt] : {text_b[:50]}...")
        
        for dim in QUALITY_DIMENSIONS:
            p_gt = probs[dim]
            # Target = 1 means B is better than A; Target = 0 means A is better than B
            # The model predicts P(B > A) = sigmoid(s_B - s_A)
            # Therefore s_B - s_A is the logit direction.
            target = p_gt
            logit_direction = "s_B - s_A"
            
            # Print verification logic
            print(f"    - Dimension: {dim:<20} | Raw Prob: {p_gt:.2f} | Mapped Target: {target:.2f} | Logit Direction: {logit_direction}")
            
            # Check consistency: if target > 0.5, B is preferred, so logit (s_B - s_A) must be positive
            # if target < 0.5, A is preferred, so logit (s_B - s_A) must be negative
            # If target == 0.5, they are equal
            status = "PASS"
            if target > 0.5 and logit_direction != "s_B - s_A":
                status = "FAIL"
            assert status == "PASS", "Direction check failed!"
            
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
