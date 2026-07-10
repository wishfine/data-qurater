import os
import json
import argparse
import csv

DIMENSIONS = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]

def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints and plot learning progress")
    parser.add_argument("--eval_dir", type=str, default="outputs/qwen35_4b_experiment/evaluations")
    parser.add_argument("--output_md", type=str, default="reports/server/training_comparison.md")
    parser.add_argument("--output_json", type=str, default="reports/server/training_comparison.json")
    parser.add_argument("--learning_curve", type=str, default="outputs/qwen35_4b_experiment/evaluations/learning_curve.csv")
    args = parser.parse_args()

    # Locate evaluation JSONs
    # Expecting: baseline_eval.json, smoke_eval.json
    baseline = load_metrics(os.path.join(args.eval_dir, "baseline_eval.json"))
    smoke = load_metrics(os.path.join(args.eval_dir, "smoke_eval.json"))

    comparison_data = {}
    if baseline:
        comparison_data["baseline"] = baseline
    if smoke:
        comparison_data["smoke"] = smoke

    # Save training_comparison.json
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    # Save training_comparison.md
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Checkpoint Training Comparison Report\n\n")
        f.write("| Dimension / Metric | Baseline (Checkpoint-0) | Smoke (Trained) |\n")
        f.write("| --- | --- | --- |\n")
        
        # 1. Macro Accuracy
        acc_base = f"{baseline['macro_accuracy']:.4f}" if baseline else "N/A"
        acc_smoke = f"{smoke['macro_accuracy']:.4f}" if smoke else "N/A"
        f.write(f"| **Macro Accuracy** | {acc_base} | {acc_smoke} |\n")
        
        # 2. Individual Dimensions Accuracies
        for dim in DIMENSIONS:
            base_val = f"{baseline[dim]['accuracy']:.4f}" if baseline and dim in baseline else "N/A"
            smoke_val = f"{smoke[dim]['accuracy']:.4f}" if smoke and dim in smoke else "N/A"
            f.write(f"| {dim} Accuracy | {base_val} | {smoke_val} |\n")
            
        # 3. BCE Loss
        for dim in DIMENSIONS:
            base_val = f"{baseline[dim]['bce_loss']:.4f}" if baseline and dim in baseline else "N/A"
            smoke_val = f"{smoke[dim]['bce_loss']:.4f}" if smoke and dim in smoke else "N/A"
            f.write(f"| {dim} BCE Loss | {base_val} | {smoke_val} |\n")

        # 4. AUC
        for dim in DIMENSIONS:
            base_val = f"{baseline[dim]['auc']:.4f}" if baseline and dim in baseline else "N/A"
            smoke_val = f"{smoke[dim]['auc']:.4f}" if smoke and dim in smoke else "N/A"
            f.write(f"| {dim} AUC | {base_val} | {smoke_val} |\n")
            
    # Save learning_curve.csv
    os.makedirs(os.path.dirname(args.learning_curve), exist_ok=True)
    with open(args.learning_curve, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "macro_accuracy", "writing_style_acc", "required_expertise_acc", "facts_and_trivia_acc", "educational_value_acc"])
        if baseline:
            writer.writerow([
                "checkpoint-0",
                baseline.get("macro_accuracy", 0.0),
                baseline.get("writing_style", {}).get("accuracy", 0.0),
                baseline.get("required_expertise", {}).get("accuracy", 0.0),
                baseline.get("facts_and_trivia", {}).get("accuracy", 0.0),
                baseline.get("educational_value", {}).get("accuracy", 0.0),
            ])
        if smoke:
            writer.writerow([
                "checkpoint-smoke",
                smoke.get("macro_accuracy", 0.0),
                smoke.get("writing_style", {}).get("accuracy", 0.0),
                smoke.get("required_expertise", {}).get("accuracy", 0.0),
                smoke.get("facts_and_trivia", {}).get("accuracy", 0.0),
                smoke.get("educational_value", {}).get("accuracy", 0.0),
            ])

    print(f"Comparison report generated: {args.output_md}")

if __name__ == "__main__":
    main()
