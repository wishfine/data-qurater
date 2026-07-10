import os
import json
import argparse
import csv
import re

DIMENSIONS = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]

def load_metrics(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_format(val):
    if val is None:
        return "N/A"
    try:
        return f"{val:.4f}"
    except (TypeError, ValueError):
        return "N/A"

def extract_epoch(filename):
    # Matches files like epoch_1.5_eval.json
    match = re.match(r"epoch_([0-9\.]+)_eval\.json", filename)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return -1.0
    # Fallbacks
    if filename == "baseline_eval.json":
        return 0.0
    if filename == "smoke_eval.json":
        return 999.0
    return -1.0

def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints and plot learning progress")
    parser.add_argument("--eval_dir", type=str, default="outputs/qwen35_4b_experiment/evaluations")
    parser.add_argument("--output_md", type=str, default="reports/server/training_comparison.md")
    parser.add_argument("--output_json", type=str, default="reports/server/training_comparison.json")
    parser.add_argument("--learning_curve", type=str, default="outputs/qwen35_4b_experiment/evaluations/learning_curve.csv")
    args = parser.parse_args()

    # Scan the evaluations directory for all files matching *.json
    eval_files = []
    if os.path.exists(args.eval_dir):
        for f in os.listdir(args.eval_dir):
            if f.endswith(".json") and f != "learning_curve.json":
                epoch_val = extract_epoch(f)
                if epoch_val >= 0.0:
                    eval_files.append((f, epoch_val))

    # Sort files chronologically by epoch
    eval_files.sort(key=lambda x: x[1])

    checkpoints_data = {}
    for filename, epoch_val in eval_files:
        path = os.path.join(args.eval_dir, filename)
        metrics = load_metrics(path)
        if metrics:
            name = filename.replace("_eval.json", "").replace("_", " ").title()
            checkpoints_data[name] = metrics

    # Save training_comparison.json
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(checkpoints_data, f, indent=2)

    # Save training_comparison.md
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Checkpoint Training Comparison Report\n\n")
        
        if not checkpoints_data:
            f.write("No evaluation files found.\n")
            return

        headers = ["Dimension / Metric"] + list(checkpoints_data.keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        # 1. Macro Accuracy
        row = ["**Macro Accuracy**"]
        for name in checkpoints_data:
            row.append(safe_format(checkpoints_data[name].get("macro_accuracy")))
        f.write("| " + " | ".join(row) + " |\n")

        # 2. Individual Dimensions Accuracies
        for dim in DIMENSIONS:
            row = [f"{dim} Accuracy"]
            for name in checkpoints_data:
                row.append(safe_format(checkpoints_data[name].get(dim, {}).get("accuracy")))
            f.write("| " + " | ".join(row) + " |\n")

        # 3. BCE Loss
        for dim in DIMENSIONS:
            row = [f"{dim} BCE Loss"]
            for name in checkpoints_data:
                row.append(safe_format(checkpoints_data[name].get(dim, {}).get("bce_loss")))
            f.write("| " + " | ".join(row) + " |\n")

        # 4. AUC
        for dim in DIMENSIONS:
            row = [f"{dim} AUC"]
            for name in checkpoints_data:
                row.append(safe_format(checkpoints_data[name].get(dim, {}).get("auc")))
            f.write("| " + " | ".join(row) + " |\n")

    # Save learning_curve.csv
    os.makedirs(os.path.dirname(args.learning_curve), exist_ok=True)
    with open(args.learning_curve, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "macro_accuracy", "writing_style_acc", "required_expertise_acc", "facts_and_trivia_acc", "educational_value_acc"])
        for name, metrics in checkpoints_data.items():
            writer.writerow([
                name.lower().replace(" ", "-"),
                metrics.get("macro_accuracy", 0.0),
                metrics.get("writing_style", {}).get("accuracy", 0.0),
                metrics.get("required_expertise", {}).get("accuracy", 0.0),
                metrics.get("facts_and_trivia", {}).get("accuracy", 0.0),
                metrics.get("educational_value", {}).get("accuracy", 0.0),
            ])

    print(f"Comparison report generated: {args.output_md}")

if __name__ == "__main__":
    main()
