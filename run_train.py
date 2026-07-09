#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Instruct LoRA/QLoRA 训练启动脚本。
您可以在服务器上运行此脚本，它会依次训练物理和化学题目的 LoRA 模型。
"""
import subprocess
import os

# 定义基础配置（请根据服务器路径进行调整）
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # 对应您提到的 ~4B 大小（Qwen2.5 系列有 3B 和 7B）
MAX_LEN = 2048
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4

# 如果您在多卡服务器上，可以使用 CUDA_VISIBLE_DEVICES 限制 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_cmd(cmd):
    print(f"正在执行命令:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

# ----------------- 1. 物理 LoRA 训练 -----------------
# 物理标注数据集（若想训练包含 18 维特征的完整打标模型，请使用 rated_results.jsonl）
# 若想只训练纯 5 档分类模型，可直接指向 data/ 目录下的 raw 题目文件
physics_train_cmd = [
    "python", "train_sft_lora.py",
    "--model_name_or_path", BASE_MODEL,
    "--train_data", "../prompt_test/physics_difficulty_rated_results.jsonl", # 指向有 GPT 标注的 dataset
    "--prompt_file", "../初中物理难度打标提示词.txt",
    "--output_dir", "./outputs/qwen_physics_lora",
    "--use_qlora",
    "--bf16",
    "--max_seq_length", str(MAX_LEN),
    "--num_train_epochs", str(EPOCHS),
    "--per_device_train_batch_size", str(BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRAD_ACCUM),
]

print("=== 开始训练初中物理难度打标 LoRA 模型 ===")
run_cmd(physics_train_cmd)


# ----------------- 2. 化学 LoRA 训练 -----------------
chemistry_train_cmd = [
    "python", "train_sft_lora.py",
    "--model_name_or_path", BASE_MODEL,
    "--train_data", "../prompt_test/chemistry_difficulty_rated_results.jsonl",
    "--prompt_file", "../初中化学难度打标提示词.txt",
    "--output_dir", "./outputs/qwen_chemistry_lora",
    "--use_qlora",
    "--bf16",
    "--max_seq_length", str(MAX_LEN),
    "--num_train_epochs", str(EPOCHS),
    "--per_device_train_batch_size", str(BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRAD_ACCUM),
]

print("=== 开始训练初中化学难度打标 LoRA 模型 ===")
run_cmd(chemistry_train_cmd)

print("=== 所有科目 LoRA 训练完成！ ===")
