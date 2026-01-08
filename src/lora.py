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
    
    return {
        "input_ids": input_ids,
        "labels": input_ids,
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

args = TrainingArguments(
    output_dir="./output/Qwen2.5_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=10,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-fintune",
    experiment_name="Qwen2.5-1.5B-lora-fintune",
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

swanlab.finish()
