import os
import json
import random

def generate_smoke_data():
    output_dir = "data/qurating"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4 dimensions of quality rating
    # 0: writing_style, 1: required_expertise, 2: facts_and_trivia, 3: educational_value
    dimensions = [0, 1, 2, 3]
    domains = ["general", "academic", "creative", "code"]
    
    texts_pool = [
        "Python is a high-level programming language known for its readability and clean syntax.",
        "The quick brown fox jumps over the lazy dog in a classical pangram sequence.",
        "Quantum computing operates on qubits which utilize superposition and entanglement principles.",
        "To cook a perfect omelette, whisk eggs thoroughly and cook over low heat with melted butter.",
        "Artificial Intelligence is rapidly advancing across various sectors including medicine and finance.",
        "The Great Wall of China is one of the world's most famous historical monuments.",
        "Machine learning models require robust preprocessing pipelines to clean features and scale weights.",
        "Shakespeare was a prominent English playwright who wrote famous tragedies like Hamlet and Macbeth.",
        "Deep learning leverages multi-layered neural networks to approximate complex mathematical functions.",
        "Proper hydration and sleep are essential pillars for maintaining optimal biological health."
    ]
    
    def create_records(num_records):
        records = []
        for _ in range(num_records):
            text_a = random.choice(texts_pool)
            # Make sure text_b is different from text_a
            text_b = random.choice(texts_pool)
            while text_b == text_a:
                text_b = random.choice(texts_pool)
                
            records.append({
                "text_a": text_a,
                "text_b": text_b,
                "target": round(random.uniform(0.0, 1.0), 4),
                "dimension_id": random.choice(dimensions),
                "confidence": round(random.uniform(0.1, 1.0), 4),
                "domain": random.choice(domains)
            })
        return records

    # Write smoke_train.jsonl (64 samples)
    smoke_train_path = os.path.join(output_dir, "smoke_train.jsonl")
    train_records = create_records(64)
    with open(smoke_train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {smoke_train_path} with 64 records.")
    
    # Write smoke_eval.jsonl (16 samples)
    smoke_eval_path = os.path.join(output_dir, "smoke_eval.jsonl")
    eval_records = create_records(16)
    with open(smoke_eval_path, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {smoke_eval_path} with 16 records.")

if __name__ == "__main__":
    generate_smoke_data()
