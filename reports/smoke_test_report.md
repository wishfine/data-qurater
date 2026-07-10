# QuRater Smoke Test Benchmark Report

This report records the setup, verification details, and benchmarking metrics for the QwenQuRater smoke test.

## 1. Dry-Run Setup
The smoke test is configured via `scripts/run_qurater_smoke.sh` using:
* **Dataset**: `example_train_data.jsonl` (contains 8 pairwise samples comparing text A and B).
* **Sample Bounds**: `--max_train_samples 8 --max_eval_samples 8`.
* **Batch Size**: 2.
* **Epochs**: 1.
* **Quantization**: Enabled (NF4 4-bit loading of the backbone).
* **Tuning**: LoRA (r=8, alpha=16) targeting dynamic projection and FFN layers.

## 2. Pre-execution Code Verification
Before launching the model weights on GPU, the pipeline automatically runs the following self-verifications:
1. **Bradley-Terry Direction Check**:
   * Evaluates $\text{loss}_1$ when $s_B > s_A$ for target $y=1.0$ ($B \succ A$).
   * Evaluates $\text{loss}_2$ when $s_A > s_B$ for target $y=1.0$ ($B \succ A$).
   * Asserts $\text{loss}_1 < \text{loss}_2$ to guarantee that positive updates push preferred text scores higher and non-preferred scores lower.
   * *Status*: Passed successfully in unit compilation.

2. **Dimension Mapping Verification**:
   * Confirms `facts_trivia` in raw datasets is correctly mapped to `facts_and_trivia`.

## 3. Benchmarking Framework (Pending Conda Env Installation)
Once the conda environment finishes installing and the primary Qwen3.5-4B model is loaded, the smoke test will measure and record:
* **GPU Memory Footprint**: Initial base allocation and peak training memory.
* **Model Step Latency**:
  * Forward pass latency (secs).
  * Backward pass latency (secs).
  * Optimizer step latency (secs).
  * Overall iteration throughput (tokens/sec).
* **Qwen3.5 HLA / DeltaNet pathway check**: Logs whether Hybrid Linear Attention or Gated DeltaNet leverages fast CUDA kernels (`fla` library) or degrades to native PyTorch fallback loops.
* **Accuracy Metrics**: Basic accuracy scores on the 8 validation items.
