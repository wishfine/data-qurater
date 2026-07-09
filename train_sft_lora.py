#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5/Qwen2.5-Instruct LoRA/QLoRA 训练脚本
用于物理和化学题目的难度及特征结构化打标微调。
支持：
  1. 从提示词文件动态加载系统 prompt。
  2. 兼容包含完整 difficulty_rating JSON 的标注数据集和仅有整数 difficulty 的原始数据集。
  3. 支持 ModelScope (魔搭社区) 自动下载，完美支持国内服务器部署。
  4. QLoRA (4-bit 压缩量化) 及 FP16/BF16 混合精度。
  5. 兼容新老版本 TRL (支持新版 SFTConfig 以及老版 TrainingArguments/max_seq_length 参数，解决 SFTTrainer 传参报错)。
"""

import os
import json
import sys
import torch
import argparse
from typing import Dict, Any, List
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

# 1. 尝试导入 SFTConfig 并做 Fallback 兼容
try:
    from trl import SFTConfig
    HAS_SFT_CONFIG = True
    print("系统: 成功导入 SFTConfig。将使用新版 SFTConfig 进行参数配置。")
except ImportError:
    from transformers import TrainingArguments as SFTConfig
    HAS_SFT_CONFIG = False
    print("系统: 未找到 SFTConfig，降级使用 TrainingArguments 配合 SFTTrainer 进行配置。")

# 2. 尝试导入 DataCollatorForCompletionOnlyLM (新版本 TRL 0.20+ 已将其移除，改用 Fallback 自定义实现)
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    print("提示: 检测到当前运行环境的 TRL 版本已移除了 DataCollatorForCompletionOnlyLM，将启用自定义 Fallback Collator。")
    
    class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        """
        自定义 Fallback 数据整理器。
        继承自 DataCollatorForLanguageModeling，但绕过其内部容易出错的 tokenizer.pad(labels) 逻辑，
        直接手动对 input_ids、attention_mask 和 labels 进行 Padding，确保维度形状绝对一致。
        """
        def __init__(self, response_template: str, tokenizer, *args, **kwargs):
            super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
            self.response_template = response_template
            # 提取响应模板在分词后的 Token IDs
            self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
            # 备用：不包含换行的前缀
            alt_template = response_template.rstrip()
            self.alt_token_ids = tokenizer.encode(alt_template, add_special_tokens=False)

        def torch_call(self, examples):
            # 1. 过滤掉无法被转化为 Tensor 的非模型输入字段 (例如 'messages')
            examples = [{k: v for k, v in ex.items() if k != "messages"} for ex in examples]
            
            # 2. 收集原始序列（确保将所有 numpy array/torch tensor 归一化为原生 python list 避免类型冲突）
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for ex in examples:
                ids = ex["input_ids"]
                if torch.is_tensor(ids):
                    ids = ids.tolist()
                batch_input_ids.append(ids)
                
                mask = ex.get("attention_mask", [1] * len(ids))
                if torch.is_tensor(mask):
                    mask = mask.tolist()
                batch_attention_mask.append(mask)
                
                lbl = ex.get("labels", ids)
                if torch.is_tensor(lbl):
                    lbl = lbl.tolist()
                batch_labels.append(lbl)
                
            # 3. 确定 Batch 最大序列长度
            max_len = max(len(ids) for ids in batch_input_ids)
            
            # 4. 手动对其进行 Padding (与 Qwen 一致，使用 right padding)
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            
            for ids, mask, lbl in zip(batch_input_ids, batch_attention_mask, batch_labels):
                remainder = max_len - len(ids)
                if padding_side == "left":
                    padded_input_ids.append([pad_token_id] * remainder + ids)
                    padded_attention_mask.append([0] * remainder + mask)
                    padded_labels.append([-100] * remainder + lbl)
                else:
                    padded_input_ids.append(ids + [pad_token_id] * remainder)
                    padded_attention_mask.append(mask + [0] * remainder)
                    padded_labels.append(lbl + [-100] * remainder)
                    
            # 5. 执行 Completion-Only Mask 遮罩逻辑
            for i in range(len(examples)):
                ids = batch_input_ids[i] # 基于未填充的原始序列定位模板，避免填充偏移
                idx = -1
                n_template = len(self.response_token_ids)
                for j in range(len(ids) - n_template + 1):
                    if ids[j : j + n_template] == self.response_token_ids:
                        idx = j + n_template
                        break
                        
                # 备用匹配不含换行的前缀
                if idx == -1:
                    n_alt = len(self.alt_token_ids)
                    for j in range(len(ids) - n_alt + 1):
                        if ids[j : j + n_alt] == self.alt_token_ids:
                            idx = j + n_alt
                            break
                            
                if idx != -1:
                    # 如果找到了模板，将模板之前（Prompt部分）的 Label 设为 -100
                    remainder = max_len - len(ids)
                    offset = remainder if padding_side == "left" else 0
                    for j in range(offset + idx):
                        padded_labels[i][j] = -100
                else:
                    print(f"警告: 样本 {i} 中未检测到助理回复模板 '{self.response_template.strip()}'，跳过 Loss 屏蔽...")
                    
            # 6. 打包并返回 PyTorch Tensor 字典
            batch = {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
            }
            return batch

# 难度级别映射表
LEVEL_MAP = {
    1: "送分题",
    2: "基础题",
    3: "中等题",
    4: "拔高题",
    5: "压轴题",
}

def load_system_prompt(prompt_path: str) -> str:
    """加载并解析提示词文件，提取前缀作为 System Prompt"""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"找不到提示词文件：{prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 兼容 python 变量格式与纯文本分割格式
    try:
        namespace = {}
        exec(content, namespace)
        prefix = namespace.get("DIFFICULTY_RATING_PROMPT_PREFIX")
        if prefix:
            return str(prefix).strip()
    except Exception:
        pass
    
    if "## 输入题目信息" in content:
        return content.split("## 输入题目信息")[0].strip()
    
    return content.strip()

def construct_question_content(data: Dict[str, Any]) -> str:
    """拼装单道题目的输入文本，兼容子题逻辑"""
    parts = []
    stem = str(data.get("stem", "") or "").strip()
    options = str(data.get("options", "") or "").strip()
    analysis = str(data.get("analysis", "") or "").strip()

    if stem:
        parts.append(f"【题干】\n{stem}")
    if options:
        parts.append(f"【选项】\n{options}")
    if analysis:
        parts.append(f"【解析】\n{analysis}")

    sub_questions = data.get("sub_questions", []) or []
    if sub_questions:
        try:
            sub_questions.sort(key=lambda x: int(x.get("question_id", 0)) if isinstance(x, dict) else 0)
        except Exception:
            pass
        parts.append("【小题】")
        for i, sq in enumerate(sub_questions, 1):
            parts.append(f"  小题{i}:")
            if isinstance(sq, dict):
                sq_stem = str(sq.get("stem", "") or "").strip()
                sq_options = str(sq.get("options", "") or "").strip()
                sq_analysis = str(sq.get("analysis", "") or "").strip()
                if sq_stem:
                    parts.append(f"    题干: {sq_stem}")
                if sq_options:
                    parts.append(f"    选项: {sq_options}")
                if sq_analysis:
                    parts.append(f"    解析: {sq_analysis}")
            else:
                parts.append(f"    题干: {sq}")

    return "\n\n".join(parts)

def process_jsonl_data(data_path: str, system_prompt: str) -> List[Dict[str, Any]]:
    """读取并处理 JSONL 格式数据集，返回标准对话格式数据集"""
    processed_samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            # 构建 User 输入
            question_content = construct_question_content(item)
            user_content = f"## 输入题目信息\n\n{question_content}\n\n请根据以上信息，对题目进行全面的难度分析和评级。"
            
            # 构建 Assistant 预期输出 (目标 JSON)
            # 1. 优先使用 GPT 已经标注好的完整 difficulty_rating
            if "difficulty_rating" in item and isinstance(item["difficulty_rating"], dict):
                assistant_content = json.dumps(item["difficulty_rating"], ensure_ascii=False)
            # 2. 备选：如果只有粗难度整数值，则退化为仅输出难度的简单 JSON
            elif "difficulty" in item:
                level_str = LEVEL_MAP.get(int(item["difficulty"]), "中等题")
                fallback_rating = {"difficulty_level": level_str}
                assistant_content = json.dumps(fallback_rating, ensure_ascii=False)
            else:
                continue  # 无效样本，跳过
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
            processed_samples.append({"messages": messages})
            
    print(f"成功加载并处理了 {len(processed_samples)} 条来自 {data_path} 的数据。")
    return processed_samples

def train():
    parser = argparse.ArgumentParser(description="Qwen SFT LoRA/QLoRA Fine-tuning")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B",
                        help="Hugging Face / ModelScope 模型 ID，或本地路径")
    parser.add_argument("--model_cache_dir", type=str, default="/home/zhangyonglin/models",
                        help="模型缓存目录")
    parser.add_argument("--use_modelscope", type=str, default="true", choices=["true", "false"],
                        help="是否通过 ModelScope (魔搭) 自动下载，国内服务器推荐使用")
    parser.add_argument("--train_data", type=str, required=True, help="训练 JSONL 数据集路径")
    parser.add_argument("--val_data", type=str, default=None, help="可选的验证 JSONL 数据集路径")
    parser.add_argument("--prompt_file", type=str, required=True, help="对应学科的打标提示词 .txt 文件路径")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_output", help="LoRA 权重保存路径")
    
    # 显存及量化参数
    parser.add_argument("--use_qlora", action="store_true", help="是否启用 QLoRA 4-bit 量化训练以节省显存")
    parser.add_argument("--bf16", action="store_true", help="是否启用 BF16 混合精度 (需要 GPU 支持)")
    parser.add_argument("--fp16", action="store_true", help="是否启用 FP16 混合精度")
    
    # 超参数配置
    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列截断长度")
    parser.add_argument("--r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="单卡训练 Batch Size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志输出步数")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--save_steps", type=int, default=500, help="当 save_strategy=steps 时的保存步长")
    parser.add_argument("--eval_strategy", type=str, default="no", choices=["no", "epoch", "steps"])
    parser.add_argument("--eval_steps", type=int, default=500, help="当 eval_strategy=steps 时的评估步长")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()

    # 1. 动态加载学科打标的系统提示词
    print(f"正在从 {args.prompt_file} 加载系统提示词...")
    system_prompt = load_system_prompt(args.prompt_file)

    # 2. 读取并构建 Hugging Face 格式 Dataset
    train_samples = process_jsonl_data(args.train_data, system_prompt)
    train_dataset = Dataset.from_list(train_samples)
    
    val_dataset = None
    if args.val_data:
        val_samples = process_jsonl_data(args.val_data, system_prompt)
        val_dataset = Dataset.from_list(val_samples)

    # 3. 魔搭 ModelScope 模型在线/本地定位
    model_path = args.model_name_or_path
    if not os.path.exists(model_path):
        if args.use_modelscope.lower() == "true":
            print(f"检测到本地不存在路径 '{model_path}'，正在通过 ModelScope 自动下载模型到 '{args.model_cache_dir}' ...")
            try:
                from modelscope import snapshot_download
                model_path = snapshot_download(args.model_name_or_path, cache_dir=args.model_cache_dir)
                print(f"ModelScope 模型下载并定位成功，本地路径为: {model_path}")
            except Exception as e:
                print(f"ModelScope 下载失败: {e}。将退回到普通 Hugging Face 模式加载...")
        else:
            print(f"本地不存在路径 '{model_path}'，将直接使用 Hugging Face 进行在线加载...")

    # 4. 配置模型量化 (QLoRA)
    bnb_config = None
    if args.use_qlora:
        print("正在启用 QLoRA 4-bit 量化配置...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # 5. 加载 Tokenizer 与 Base Model
    print(f"正在加载基座模型：{model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 and not args.use_qlora else (torch.float16 if args.fp16 else torch.float32)
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # 6. 配置 LoRA 适配器 (针对 Qwen 系列的标准 Linear 层)
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 7. 配置仅对 Assistant 的回答进行计算的 Collator (Masked Loss)
    # 对于 ChatML 格式 of Qwen，助理的回答部分前缀为 "<|im_start|>assistant\n"
    response_template = "<|im_start|>assistant\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # 8. 区分新老版本 TRL 来构建配置参数
    # 如果是新版本（有 SFTConfig），max_seq_length 和 dataset_kwargs 应该作为 SFTConfig 参数传入
    dataset_kwargs = {
        "add_special_tokens": False, # apply_chat_template 已经处理了特殊 Token
        "truncation": True,          # 强制截断到 max_seq_length，防止长文本导致的显存 Swap 降速
        "max_length": args.max_seq_length,
    }

    if HAS_SFT_CONFIG:
        print("SFT 参数配置：使用新版 SFTConfig，max_seq_length 和 dataset_kwargs 已注入 Config 中。")
        training_args = SFTConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            seed=args.seed,
            remove_unused_columns=True,
            report_to="tensorboard" if os.path.exists("./logs") else "none",
            max_length=args.max_seq_length, # 注入 SFTConfig 字段
            dataset_kwargs=dataset_kwargs,  # 新版 TRL 需注入 SFTConfig 字段
        )
        trainer_extra_kwargs = {}
    else:
        print("SFT 参数配置：由于未找到 SFTConfig，回退至 TrainingArguments，参数将在 Trainer 中直接初始化。")
        training_args = SFTConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            seed=args.seed,
            remove_unused_columns=True,
            report_to="tensorboard" if os.path.exists("./logs") else "none"
        )
        trainer_extra_kwargs = {
            "max_seq_length": args.max_seq_length,
            "dataset_kwargs": dataset_kwargs  # 老版 TRL 允许传给 SFTTrainer 构造函数
        }

    # 动态检测 SFTTrainer 的构造参数，自适应使用 tokenizer 或 processing_class
    import inspect
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_extra_kwargs["processing_class"] = tokenizer
    else:
        trainer_extra_kwargs["tokenizer"] = tokenizer

    # 9. 使用 TRL SFTTrainer 启动训练
    print("开始初始化 SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        data_collator=data_collator,
        args=training_args,
        formatting_func=lambda example: tokenizer.apply_chat_template(example["messages"], tokenize=False),
        **trainer_extra_kwargs
    )

    print("开始训练...")
    trainer.train()

    # 10. 保存微调权重
    print(f"训练完成！正在将 LoRA 权重保存至：{args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("保存完毕。")

if __name__ == "__main__":
    train()
