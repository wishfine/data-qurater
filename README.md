# QuRater 质量评分模型实现

基于ICML 2024论文《QuRating: Selecting High-Quality Data for Training Language Models》的复现实现。

## 论文核心内容
QuRater是一个用于大语言模型预训练数据质量筛选的评分模型，从四个维度对文本质量进行量化评分：
1. **writing_style** - 写作风格：语言流畅性、逻辑性、表达规范性
2. **required_expertise** - 所需专业知识：理解文本需要的专业领域知识深度
3. **facts_trivia** - 事实与趣闻：事实性信息的准确性、丰富性、趣味性
4. **educational_value** - 教育价值：传授知识、培养技能、启发思维的价值

模型通过Bradley-Terry成对比较框架训练，学习将LLM的两两比较结果转化为绝对标量评分。

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `train_qurater.py` | QuRater模型主训练脚本 |
| `generate_pairwise_labels.py` | 调用GPT生成两两比较软标签的脚本 |
| `score_and_select.py` | 用训练好的模型对语料打分并执行softmax采样筛选 |
| `example_train_data.jsonl` | 训练数据示例，包含8对标注好的文本对 |
| `requirements.txt` | Python依赖列表 |

---

## 训练数据格式

训练数据支持jsonl格式，每行一个样本，格式如下：
```json
{
  "text_a": "待比较的第一段文本",
  "text_b": "待比较的第二段文本",
  "probs": {
    "writing_style": 0.35,       // B比A好的概率 [0,1]
    "required_expertise": 0.95,  // 0.0 = A绝对更好，1.0 = B绝对更好，0.5 = 两者相当
    "facts_trivia": 0.98,
    "educational_value": 0.92
  }
}
```

根据论文描述：
- 使用GPT-3.5对文本对在四个维度上进行两两比较
- 不是简单输出二分类标签，而是记录B优于A的置信度概率P(B>A)
- 训练集共25万对：20万对通用领域随机抽取 + 维基百科/书籍/StackExchange/GitHub/arXiv五个领域各抽1万对

---

## 完整工作流程

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. （可选）使用GPT生成标注数据
如果你有大量未标注的纯文本，可以用GPT来生成两两比较标签：
```bash
export OPENAI_API_KEY="your-api-key-here"
python generate_pairwise_labels.py \
  --input_texts your_corpus.jsonl \
  --output_file pairwise_labels.jsonl \
  --num_pairs 200000 \
  --num_workers 8 \
  --openai_model gpt-3.5-turbo
```
你也可以使用OpenAI兼容的API端点通过`--base_url`参数指定。

### 3. 训练QuRater模型
```bash
python train_qurater.py \
  --model_name_or_path princeton-nlp/Sheared-LLaMA-1.3B \
  --train_data pairwise_labels.jsonl \
  --val_data val_labels.jsonl \  # 可选验证集
  --output_dir ./qurater_model \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --learning_rate 2e-5 \
  --num_epochs 3 \
  --max_length 512
```
- 论文默认使用Sheared-LLaMA-1.3B作为基础模型，你也可以替换为其他开源模型（如Qwen2、Llama3等）
- 模型架构：基础Transformer + 4个独立线性输出头，分别预测四个维度的评分
- 训练损失：四个维度的Bradley-Terry损失取平均，使用二元交叉熵（支持软标签）
- 论文报告在预留测试集上准确率超过93%

### 4. 对大规模语料打分并筛选高质量数据
```bash
python score_and_select.py \
  --model_dir ./qurater_model \
  --input_corpus your_raw_corpus.jsonl \
  --output_file selected_high_quality_data.jsonl \
  --select_dimension educational_value \
  --n_select 3000000 \
  --temperature 1.0 \
  --batch_size 16
```
- `--select_dimension`：选择用哪个维度的分数采样，论文发现**educational_value**效果最好
- `--temperature`：softmax温度τ
  - τ→0：接近top-k选择，质量最高但多样性低
  - τ→∞：接近均匀随机采样，多样性最高但质量平均
  - 论文建议平衡质量和多样性，通常τ=1.0左右效果较好
- 输出文件会包含筛选出的文本以及四个维度的评分

---

## 模型原理：Bradley-Terry损失

对于每个文本对(A,B)，模型为每个维度预测标量评分s_A和s_B，B优于A的概率为：
$$P(B \succ A) = \sigma(s_B - s_A) = \frac{1}{1+e^{-(s_B - s_A)}}$$

训练目标是最小化预测概率与GPT给出的真实概率之间的二元交叉熵损失：
$$L = -[P_{gt} \log P_{pred} + (1-P_{gt}) \log (1-P_{pred})]$$

与传统二分类奖励模型不同，这里的标签是**软概率**而非0/1硬标签，这允许GPT表达不确定的判断，提高了标签质量。

---

## 数据选择策略

论文使用基于softmax的无放回采样方法：
$$P(d_i) = \frac{\exp(s_i/\tau)}{\sum_j \exp(s_j/\tau)}$$

这种方法相比简单的top-k筛选能更好地平衡数据质量和多样性。实验表明，用这种方法筛选30B token训练出的1.3B模型，性能相当于用均匀采样训练50%更多步数。

---

## 参考
- 论文：[QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739) (ICML 2024 Spotlight)
- 基础模型：[Sheared-LLaMA-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B)
