# QuRating Reproduction Audit Report

This document records the code audit and adaptation decisions for reproducing the **QuRating** (ICML 2024) data quality rating framework using a **Qwen3.5-4B** backbone.

## 1. Codebase Audit & Legacy Isolation
* **Legacy SFT Pipeline Removed**: All generation-based SFT models, prompt templates (such as difficulty level ratings), and secondary question/answer tags have been isolated into `legacy/difficulty_rating/`.
* **Methodology Alignment**: We no longer perform token-generation SFT on task outputs. Instead, we train a pairwise scalar rating predictor using the Bradley-Terry comparison framework.

## 2. Methodology Gap Analysis

| Feature | Paper Configuration (Original) | Local Adaptation (This Repo) |
|---|---|---|
| **Backbone Model** | Sheared-LLaMA-1.3B | Qwen3.5-4B (with Qwen3-4B fallback) |
| **Input Format** | Raw text segment encoding | Raw text segment encoding (no instruction templates) |
| **Output Head** | 1 output head predicting 4 logits | ModuleDict of 4 independent linear heads (or single head with size 4) |
| **Pooling Method** | Standard classification pooling (Last Token / EOS token) | Customizable: Last-token pooling (matching classification default) / Mean pooling |
| **Optimization Target** | Bradley-Terry Binary Cross-Entropy on soft probability judgments | Bradley-Terry BCE with Logits (`loss = BCE(sigmoid(s_B - s_A), P(B > A))`) |
| **Quantization & PEFT** | 16-bit / 8-bit full parameter | LoRA / QLoRA (4-bit quantization, gradient checkpointing) |

## 3. Checklist of Verified Components

### A. Bradley-Terry Loss Direction
* The probability that text B is better than text A is defined as:
  $$P(B \succ A) = \sigma(s_B - s_A)$$
* In our dataset loader, the ground-truth soft label $y = P(B \succ A)$.
* The logits input to the BCE loss must be calculated exactly as:
  $$\text{logit} = s_B - s_A$$
* Directional verification tests are implemented to ensure there is no sign inversion.

### B. Quality Dimensions
We map the raw dataset dimensions to our 4 target quality rating dimensions:
1. `writing_style` (smoothness, grammar, flow)
2. `required_expertise` (depth of knowledge, domain speciality)
3. `facts_and_trivia` (factual density, informational value; mapped from raw `facts_trivia` key)
4. `educational_value` (clarity, instructional/conceptual utility)

### C. LoRA Target Modules detection
* We implement dynamic module name matching for LoRA target modules to avoid hardcoded parameter projection layers.
* Projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and FFN layers (`gate_proj`, `up_proj`, `down_proj`) will be targeted for fine-tuning.
