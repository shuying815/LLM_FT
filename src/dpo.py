import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import DataCollatorWithPadding
from tqdm import tqdm

def print_trainable_parameters(model):
    total_params = 0
    total_trainable_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()  # 计算参数数量
        total_params += param_count
        if param.requires_grad:
            total_trainable_params += param_count
            print(f"{name}: {param_count} trainable")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {total_trainable_params}")
    return total_trainable_params
    
# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("qwen/Qwen2.5-1.5B-Instruct", cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2___5-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2___5-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

if tokenizer.bos_token is None:   # qwen没有bos_token，要设置一下，不然dpo train时会报错。
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# 加载、处理数据集和测试集
TRAIN_FILE = "./train.jsonl"
VAL_FILE = "./val.jsonl"
print("正在加载并处理数据集...")

import random
from datasets import Dataset

# -------------------------
# 1. 读取全类别
# -------------------------
def get_all_classes(path):
    classes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for msg in item["messages"]:
                if msg["role"] == "assistant":
                    classes.add(msg["content"].strip())
                    break
    return sorted(list(classes))


# -------------------------
# 2. 提取 prompt / label
# -------------------------
def extract_label(item):
    for msg in item["messages"]:
        if msg["role"] == "assistant":
            return msg["content"].strip()
    return None

def extract_prompt(item):
    for msg in item["messages"]:
        if msg["role"] == "user":
            return msg["content"].strip()
    return ""


# -------------------------
# 3. 构建 DPO 数据
# -------------------------

def build_dpo_data(input_file, all_classes):
    dpo_list = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            label = extract_label(item)
            prompt = extract_prompt(item)

            negatives = [c for c in all_classes if c != label]
            neg_samples = random.sample(negatives, 3)

            for neg in neg_samples:
                dpo_list.append({
                    "prompt": prompt,
                    "chosen": label,
                    "rejected": neg
                })

    return dpo_list

# -------------------------
# 4. 主流程
# -------------------------

all_classes = get_all_classes(TRAIN_FILE)
print("类别数：", len(all_classes))
dpo_list = build_dpo_data(TRAIN_FILE, all_classes)
dpo_dataset = Dataset.from_list(dpo_list)


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)
print_trainable_parameters(model)  

training_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,

    # ⭐ DPO 核心参数
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    #max_target_length=256,

    report_to=["swanlab"],
    remove_unused_columns=False,
)


swanlab_callback = SwanLabCallback(
    project="Qwen2.5-fintune",
    experiment_name="Qwen2.5-1.5B-dpo-fintune",
    description="使用通义千问Qwen2-1.5B-Instruct模型在数据集上微调。",
    config={
        "model": "qwen/Qwen2.5-1.5B-Instruct",
        "dataset": "./train.jsonl",
    }
)

# 创建 DPOTrainer 时，直接使用 processing_class
trainer = DPOTrainer(
    model=model,
    ref_model=None,          # LoRA 场景必须 None
    args=training_args,
    train_dataset=dpo_dataset,
    callbacks=[swanlab_callback],
)

trainer.train()


VAL_FILE = "./val.jsonl"
RESULT_FILE = ".dpo_result.jsonl"
MAX_NEW_TOKENS = 12 
model.eval() # 切换到评估模式

print(f"正在读取验证集: {VAL_FILE}")
data_samples = []
with open(VAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data_samples.append(json.loads(line))

print(f"共加载 {len(data_samples)} 条验证数据。")
def predict(messages, model, tokenizer):
    """
    生成回复并只返回生成的文本部分
    """
    # 构造 Prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
   
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.1,  # 验证时温度设低一点，保证结果确定性
            top_p=0.9
        )

    # 裁剪掉 Input 部分，只取 Output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
 
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def check_correctness(response, ground_truth):
    """
    判别逻辑：只要生成的回答包含完整的 ground truth，即为正确
    """
    # 简单的字符串包含判断
    return ground_truth in response


base_correct_count = 0
ft_correct_count = 0
total_count = 0
results_log = []

print("\n🚀 开始自动化评估...")
pbar = tqdm(data_samples, desc="Evaluating", unit="sample")

for sample in pbar:
    total_count += 1
    
    # 提取输入和标准答案
    input_messages = sample["messages"][:-1]
    ground_truth = sample["messages"][-1]["content"]
    ft_response = predict(input_messages, model, tokenizer)
    is_ft_correct = check_correctness(ft_response, ground_truth)
    if is_ft_correct:
        ft_correct_count += 1

    results_log.append({
        "input": input_messages[-1]["content"], 
        "ground_truth": ground_truth,
        "ft_response": ft_response,
        "ft_correct": is_ft_correct
    })

    current_ft_acc = ft_correct_count / total_count
    pbar.set_postfix({ 
        "FT_Acc": f"{current_ft_acc:.2%}"
    })

final_ft_acc = ft_correct_count / total_count

print("\n" + "="*50)
print("最终评估报告")
print("="*50)
print(f"验证集总数: {total_count}")
print("判别标准: 生成内容必须包含 Ground Truth")
print("-" * 30)
print(f"微调模型 正确数: {ft_correct_count}")
print(f"微调模型 准确率: {final_ft_acc:.2%}")
print("="*50)

# 保存详细结果
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(results_log, f, ensure_ascii=False, indent=2)
print(f"详细对比日志已保存至: {RESULT_FILE}")

swanlab.finish()
