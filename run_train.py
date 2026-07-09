#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5 LoRA/QLoRA 训练启动脚本。
您可以在服务器上运行此脚本，它会加载您通过 ModelScope 下载至 /home/zhangyonglin/models 的 Qwen/Qwen3.5-4B 基座模型，
并依次训练物理和化学题目的 LoRA 模型。
"""
import subprocess
import os

# 定义基础配置
BASE_MODEL = "Qwen/Qwen3.5-4B"  # 已修改为您实际下载的魔搭社区模型 ID (Qwen/Qwen3.5-4B)
CACHE_DIR = "/home/zhangyonglin/models"  # 指定的服务器模型缓存目录
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
physics_train_cmd = [
    "python", "train_sft_lora.py",
    "--model_name_or_path", BASE_MODEL,
    "--model_cache_dir", CACHE_DIR,
    "--use_modelscope", "true",
    "--train_data", "data/physics_difficulty_rated_results.jsonl", # 现已移动至项目目录内
    "--prompt_file", "prompts/初中物理难度打标提示词.txt",            # 现已移动至项目目录内
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
    "--model_cache_dir", CACHE_DIR,
    "--use_modelscope", "true",
    "--train_data", "data/chemistry_difficulty_rated_results.jsonl", # 现已移动至项目目录内
    "--prompt_file", "prompts/初中化学难度打标提示词.txt",            # 现已移动至项目目录内
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
