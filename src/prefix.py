import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from datasets import load_dataset

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


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response
    
# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("qwen/Qwen2.5-1.5B-Instruct", cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2___5-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2___5-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.config.use_cache=False
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 加载、处理数据集和测试集
TRAIN_FILE = "./train.jsonl"
VAL_FILE = "./val.jsonl"
print("正在加载并处理数据集...")
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
column_names = dataset["train"].column_names
tokenized_dataset = dataset.map(
    process_func,
    batched=False,
    remove_columns=column_names,
    desc="Running tokenizer"
)


peft_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=16,   # prefix 长度，可选 16/32/64
    inference_mode=False
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

args = TrainingArguments(
    output_dir="./output/Qwen2.5_prefix",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=10,
    save_steps=1020,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=False,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-fintune",
    experiment_name="Qwen2.5-1.5B-prefix",
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
VAL_FILE = "./val.jsonl"
RESULT_FILE = "lora_result.jsonl"
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
