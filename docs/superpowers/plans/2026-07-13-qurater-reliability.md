# QuRater Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training recovery, data auditing, configuration, metrics, and server execution reliable for the Qwen QuRater pipeline.

**Architecture:** Centralize run paths and configuration in the trainer, persist resumable sampler progress in checkpoints, and move server preflight checks into deterministic scripts. Keep DDP training free of single-rank evaluation; final evaluation remains a post-training process.

**Tech Stack:** Python 3.11, PyTorch DDP, Transformers, PEFT, JSON configuration, Bash.

## Global Constraints

- The local machine has no PyTorch runtime; use static checks locally and server commands for runtime verification.
- GitHub is the only code synchronization path to the server.
- Preserve the existing modular checkpoint format.

---

### Task 1: Configuration and run paths

**Files:**
- Modify: `train_qurater_qwen.py`
- Modify: `scripts/server_run_train.sh`
- Test: `tests/test_qurater.py`

- [ ] Add a JSON config loader whose explicit CLI arguments override configuration values.
- [ ] Derive metadata and checkpoint-0 paths from `output_dir` instead of a hard-coded experiment directory.
- [ ] Add a test for derived experiment paths and configuration merging.

### Task 2: Checkpoint recovery

**Files:**
- Modify: `train_qurater_qwen.py`
- Test: `tests/test_qurater.py`

- [ ] Persist epoch, next batch index, optimizer step, and sampler seed in `trainer_state.pt`.
- [ ] Resume at the first unprocessed batch and preserve the optimizer-step counter.
- [ ] Add a test for checkpoint-state serialization and resume cursor interpretation.

### Task 3: Data and metrics integrity

**Files:**
- Modify: `data/qurating_dataset.py`
- Modify: `evaluate_qurater.py`
- Modify: `scripts/check_train_eval_overlap.py`
- Modify: `scripts/server_run_train.sh`
- Test: `tests/test_qurater.py`

- [ ] Validate normalized labels, confidence, and dimensions before batching.
- [ ] Emit all-sample and training-threshold metric summaries.
- [ ] Audit the full training/evaluation split before launching DDP.

### Task 4: DDP and server operations

**Files:**
- Modify: `train_qurater_qwen.py`
- Modify: `scripts/server_check_env.sh`
- Modify: `scripts/server_run_train.sh`
- Test: `tests/test_qurater.py`

- [ ] Guard a two-process launch with a two-GPU preflight check.
- [ ] Remove unsafe in-DDP full evaluation from the standard server path.
- [ ] Reduce routine log volume and ensure process-group cleanup.

### Verification

- [ ] Run Python compilation, Bash syntax checks, and static audit locally.
- [ ] Run smoke split leakage audit locally.
- [ ] Run `python -m unittest discover -s tests -p "test_*.py" -v` on the server after pulling the commit.
