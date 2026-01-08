from qwen2.modeling_qwen2 import Qwen2ForCausalLM
from qwen2.configuration_qwen2 import Qwen2Config
from modelscope import AutoTokenizer
import json
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from swanlab.integration.huggingface import SwanLabCallback
import os
import swanlab
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 加载预训练模型
config = Qwen2Config.from_json_file("./qwen/Qwen2___5-1___5B-Instruct/config.json")
model = Qwen2ForCausalLM(config)
model_weights = load_file("./qwen/Qwen2___5-1___5B-Instruct/model.safetensors")
model.load_state_dict(model_weights, strict=False)
model = model.to(dtype=torch.bfloat16, device="cuda")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2___5-1___5B-Instruct/", use_fast=False, trust_remote_code=True)

# 打印所有可训练参数以及可训练参数量
def print_trainable_parameters(model):
    total_params = 0
    total_trainable_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()  # 计算参数数量
        total_params += param_count
        if 'adapter' in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
        model.lm_head.weight.requires_grad = True
        if param.requires_grad:
            total_trainable_params += param_count
            print(f"{name}: {param_count} trainable")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {total_trainable_params}")
    return total_trainable_params

# 指定adapter微调
print_trainable_parameters(model)
print(model.model.forward.__code__.co_varnames) # 前向传播参数

# 数据处理函数
def process_func(example):
    MAX_LENGTH = 512
    
    input_ids = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        max_length=MAX_LENGTH,
        truncation=True
    )
    
    labels = input_ids.copy()
    user_len = len(
        tokenizer.apply_chat_template(
            example["messages"][:2],
            tokenize=True,
            add_generation_prompt=False,
            max_length=MAX_LENGTH,
            truncation=True
        )
    )

    labels[:user_len] = [-100] * user_len
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

# 加载训练和验证数据
TRAIN_FILE = "./train_reasoning.jsonl"
VAL_FILE = "./val_reasoning.jsonl"
print("正在加载并处理数据集...")
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
column_names = dataset["train"].column_names
tokenized_dataset = dataset.map(
    process_func,
    batched=False,
    remove_columns=column_names,
    desc="Running tokenizer"
)

# 训练参数
output_dir="./output/Qwen2.5_adapter_10"
args = TrainingArguments(
    output_dir="./output/Qwen2.5_adapter_10",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="steps",  # Enable saving at regular steps
    save_steps=1020,  # Save every 100 steps
    learning_rate=1e-4,
    save_on_each_node=True,  # Ensure saving on each node in multi-node training
    gradient_checkpointing=True,
    report_to="none",
)


swanlab_callback = SwanLabCallback(
    project="Qwen2.5-fintune",
    experiment_name="Qwen2.5-1.5B-Adapter-10",
    description="使用通义千问Qwen2-1.5B-Instruct模型在数据集上微调。",
    config={
        "model": "qwen/Qwen2.5-1.5B-Instruct",
        "dataset": "./train.jsonl",
    }
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)

trainer.train()


# 验证集评估
RESULT_FILE = "result.jsonl"
MAX_NEW_TOKENS = 512
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
