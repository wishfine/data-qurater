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

# 单卡 debug 训练速度判断标准：
# 1 step < 30 秒：可接受
# 1 step 30~60 秒：偏慢，但可以继续优化
# 1 step > 120 秒：仍然有严重问题

DEBUG = True  # 设置为 True 启用快速 debug 验证模型速度，设置为 False 运行完整微调训练

# 根据模式设置不同的训练参数
if DEBUG:
    MAX_LEN = 1024
    BATCH_SIZE = 2
    GRAD_ACCUM = 8
    EPOCHS = 1
    SAVE_STRATEGY = "steps"
    EVAL_STRATEGY = "no"
    LOGGING_STEPS = 1
    PROMPT_MODE = "compact"
    MAX_TRAIN_SAMPLES = 64
else:
    MAX_LEN = 1024  # 默认使用 1024，如果想调大可以改到 1536
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    EPOCHS = 3
    SAVE_STRATEGY = "steps"
    EVAL_STRATEGY = "no"
    LOGGING_STEPS = 10
    PROMPT_MODE = "compact"
    MAX_TRAIN_SAMPLES = None

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
    "--save_strategy", SAVE_STRATEGY,
    "--eval_strategy", EVAL_STRATEGY,
    "--logging_steps", str(LOGGING_STEPS),
    "--prompt_mode", PROMPT_MODE,
]
if MAX_TRAIN_SAMPLES is not None:
    physics_train_cmd.extend(["--max_train_samples", str(MAX_TRAIN_SAMPLES)])

print("=== 开始训练初中物理难度打标 LoRA 模型 ===")
run_cmd(physics_train_cmd)


# ----------------- 2. 化学 LoRA 训练 -----------------
# chemistry_train_cmd = [
#     "python", "train_sft_lora.py",
#     "--model_name_or_path", BASE_MODEL,
#     "--model_cache_dir", CACHE_DIR,
#     "--use_modelscope", "true",
#     "--train_data", "data/chemistry_difficulty_rated_results.jsonl", # 现已移动至项目目录内
#     "--prompt_file", "prompts/初中化学难度打标提示词.txt",            # 现已移动至项目目录内
#     "--output_dir", "./outputs/qwen_chemistry_lora",
#     "--use_qlora",
#     "--bf16",
#     "--max_seq_length", str(MAX_LEN),
#     "--num_train_epochs", str(EPOCHS),
#     "--per_device_train_batch_size", str(BATCH_SIZE),
#     "--gradient_accumulation_steps", str(GRAD_ACCUM),
#     "--save_strategy", SAVE_STRATEGY,
#     "--eval_strategy", EVAL_STRATEGY,
#     "--logging_steps", str(LOGGING_STEPS),
#     "--prompt_mode", PROMPT_MODE,
# ]
# if MAX_TRAIN_SAMPLES is not None:
#     chemistry_train_cmd.extend(["--max_train_samples", str(MAX_TRAIN_SAMPLES)])

# print("=== 开始训练初中化学难度打标 LoRA 模型 ===")
# run_cmd(chemistry_train_cmd)

print("=== 物理科目 LoRA 训练已完成！(已跳过化学训练) ===")
