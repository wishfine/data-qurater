# Qwen-3.5-4B Adaptation Report for QuRating

This report details the architectural adaptation of the QuRating methodology to the **Qwen3.5-4B** backbone, clarifying why it qualifies as a reproduction, highlighting differences from the original paper, and setting up fallback criteria for Qwen3-4B.

## 1. Original vs. Local Backbone

* **Original Paper Backbone**: Sheared-LLaMA-1.3B (a pruned LLaMA variant).
* **Current Backbones**:
  * **Primary**: Qwen3.5-4B (Qwen2.5 / Qwen3.5 variant).
  * **Fallback**: Qwen3-4B.

## 2. Why this is a QuRating Methodological Reproduction
Although the base model differs, the **methodology** remains 100% aligned with QuRating:
1. **Four Quality Dimensions**: Scores are output along `writing_style`, `required_expertise`, `facts_and_trivia`, and `educational_value`.
2. **Pairwise Comparison Formulation**: Learning is driven by comparing text $A$ and text $B$ rather than absolute scoring prompts.
3. **Bradley-Terry Preference Probability**: Probability estimation uses the difference in predicted scalar scores: $P(B \succ A) = \sigma(s_B - s_A)$.
4. **Soft Cross-Entropy Loss**: Optimization is performed via Binary Cross Entropy using soft judgment margins in $[0.0, 1.0]$.
5. **No Instruction templates**: Raw texts are fed directly into the model to capture the baseline text quality without prompt-induced biases.

## 3. Deviations and Potential Impacts

* **Parameter Scale**: 4B parameters vs. 1.3B. The larger capacity of Qwen3.5-4B is expected to increase rating accuracy and domain resilience.
* **Attention Mechanism**: Qwen3.5 features special linear attention components (Hybrid Linear Attention / Gated DeltaNet). This requires checking GPU fast path compatibility.
* **Tuning Method**: The original paper used full parameter fine-tuning. Due to memory and compute constraints of a 4B parameter model, we default to LoRA / QLoRA parameter-efficient training.

## 4. Qwen3-4B Fallback Criteria
If and only if all of the following conditions are met, we will fallback to **Qwen3-4B**:
1. The Conda environment installation is fully complete.
2. `torch.cuda.is_available()` returns `True`.
3. Qwen3.5-4B loads successfully but fails to leverage Hybrid Linear Attention / Gated DeltaNet fast path (i.e. slow fallback to torch).
4. Training step time exceeds a reasonable threshold (e.g. > 120s/step).
5. The slow training is verified NOT to be caused by dataloader, batch size, gradient accumulation, or sequence length.
6. The user explicitly approves the fallback choice after receiving our detailed error diagnostic report.
