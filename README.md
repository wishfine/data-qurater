# QuRating Methodology Reproduction with Qwen3-0.6B

This repository contains the reproduction of the data quality rating framework **QuRating** (ICML 2024 Spotlight), adapted to a **Qwen3-0.6B** backbone.

---

## 1. Quality Dimensions

We score text across four distinct quality dimensions:
1. **writing_style**: Fluency, coherence, vocabulary choice, and syntax.
2. **required_expertise**: Domain specificity and depth of knowledge required to comprehend the text.
3. **facts_and_trivia**: Density of factual information, events, data, and details.
4. **educational_value**: Pedagogical utility, clarity of explanations, and capacity to build skills or prompt critical thinking.

---

## 2. Bradley-Terry Preference Framework

Pairwise preferences are modeled by assigning scalar ratings $s_A$ and $s_B$. The predicted probability that $B$ is superior to $A$ in dimension $d$ is:

$$P(B \succ A) = \sigma(s_B - s_A) = \frac{1}{1 + e^{-(s_B - s_A)}}$$

We train using a soft Binary Cross-Entropy (BCE) loss against the target judgments $y = P_{gt}(B \succ A) \in [0, 1]$:

$$L = -[y \log \hat{y} + (1 - y) \log (1 - \hat{y})]$$

---

## 3. Directory Layout

```text
data-qurater/
├── models/
│   └── qwen_qurater.py          # QwenQuRater Model definition (Scheme A, last token pooling)
├── data/
│   └── qurating_dataset.py      # Pairwise label dataset loader & official adapter
├── train_qurater_qwen.py        # Pairwise trainer (BF16 LoRA, checkpoint-0 baseline)
├── evaluate_qurater.py          # Metric aggregator (accuracy, balanced acc, BCE, AUC, buckets)
├── score_corpus.py              # Document scorer with length-weighted sliding window
├── compare_checkpoints.py       # Checkpoint comparisons & learning curve exporter
├── configs/
│   ├── qwen3_06b_smoke.json     # Smoke test configuration
│   └── qwen3_06b_train.json     # Full training configuration
├── scripts/
│   ├── build_smoke_split.py     # Partition data into disjoint train/eval files via connected components
│   ├── check_train_eval_overlap.py # Audit train/eval splits for leakage
│   ├── check_environment_status.py # Check stored env status and verify live environment properties
│   ├── server_download_model.py # ModelScope download script
│   ├── server_download_model.sh # Script to download Qwen3-0.6B from ModelScope
│   ├── server_verify_model_path.sh # Lightweight model path validation script
│   ├── server_check_env.sh      # Script to verify server CUDA and library dependencies
│   ├── server_run_unit_tests.sh # Script to execute unit tests
│   ├── server_verify_data.sh    # Script to check data format, alignment and leakage (rebuilds splits)
│   ├── server_run_baseline_eval.sh  # Script to evaluate baseline (checkpoint-0)
│   ├── server_run_smoke.sh      # Script to run 2-step training smoke test
│   ├── server_test_checkpoint.sh # Script to evaluate reloaded smoke checkpoint
│   └── server_collect_report.sh # Script to aggregate all logs into summary
```

---

## 4. Checkpoint Format

Checkpoints are saved under the following modular structure:
```text
checkpoint-final/
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── rating_head.safetensors
├── qurater_config.json
├── tokenizer/
├── training_args.json
└── trainer_state.pt
```

---

## 5. Multi-Stage Execution Workflow on Target Server

Once pushed to GitHub, pull the changes on the target server, activate the `agentgym` Conda environment, and run the following verification stages sequentially.

### Phase 1: Environment Verification
```bash
cd ~/data-qurater
git pull origin main
conda activate agentgym

# If modelscope is missing, install it first (do NOT upgrade PyTorch):
python -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# Verify server hardware, dependencies, and environment status
bash scripts/server_check_env.sh
```

### Phase 2: Model Acquisition & Path Validation
*Proceed only if Phase 1 reports status PASS.*
```bash
# Download base model from ModelScope and verify path config
bash scripts/server_download_model.sh
bash scripts/server_verify_model_path.sh
```

### Phase 3: Data Separation & Verification
*Proceed only if Phase 2 passes verification.*
```bash
# Rebuild splits by connected components (无向图连通分量) and audit train-eval leakage
bash scripts/server_verify_data.sh
```

### Phase 4: Unit Testing & Baseline Evaluation
*Proceed only if Phase 3 data verification passes.*
```bash
# Run unit tests and save untrained baseline evaluation results
bash scripts/server_run_unit_tests.sh
bash scripts/server_run_baseline_eval.sh "$(cat outputs/model_path.txt)"
```

### Phase 5: Smoke Training Benchmark
```bash
# Run 2-step single GPU training benchmark (BF16 LoRA)
bash scripts/server_run_smoke.sh "$(cat outputs/model_path.txt)"
```

### Phase 6: Checkpoint Load & Evaluation comparative audit
```bash
# Load reloaded smoke checkpoint, run round-trip, and compare metrics
bash scripts/server_test_checkpoint.sh "$(cat outputs/model_path.txt)"
bash scripts/server_collect_report.sh
```

---

## 6. References
* Paper: [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739) (ICML 2024)
