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
from transformers import DataCollatorWithPadding, Trainer
from tqdm import tqdm
import torch.nn.functional as F

class SFTWithDPOTrainer(Trainer):
    def __init__(self, dpo_beta=0.1, dpo_lambda=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_beta = dpo_beta
        self.dpo_lambda = dpo_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        inputs:
        {
          input_ids, attention_mask, labels,
          prompt, chosen, rejected
        }
        """

        # ---------
        # 1. SFT loss（主损失）
        # ---------
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        sft_loss = outputs.loss
        # ---------
        # 2. DPO loss（正则项）
        # ---------
        with torch.no_grad():
            prompt_ids = inputs["prompt_ids"]
            chosen_ids = inputs["chosen_ids"]
            rejected_ids = inputs["rejected_ids"]

        # 拼 prompt + answer
        def get_logp(answer_ids):
            full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            attn = torch.ones_like(full_ids)

            out = model(full_ids, attention_mask=attn)
            logits = out.logits[:, :-1, :]
            labels = full_ids[:, 1:]

            log_probs = F.log_softmax(logits, dim=-1)
            token_logp = torch.gather(
                log_probs,
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)

            # 只算 answer 部分
            return token_logp[:, -answer_ids.size(1):].sum(dim=1)

        logp_chosen = get_logp(chosen_ids)
        logp_rejected = get_logp(rejected_ids)

        dpo_loss = -torch.mean(
            F.logsigmoid(self.dpo_beta * (logp_chosen - logp_rejected))
        )

        # ---------
        # 3. 总损失
        # ---------
        loss = sft_loss + self.dpo_lambda * dpo_loss

        return (loss, outputs) if return_outputs else loss



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
#model_dir = snapshot_download("qwen/Qwen2.5-1.5B-Instruct", cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("../qwen/qwen/Qwen2___5-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../qwen/qwen/Qwen2___5-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

if tokenizer.bos_token is None:   # qwen没有bos_token，要设置一下，不然dpo train时会报错。
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

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

from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("./paraphrase-multilingual-MiniLM-L12-v2")

from functools import partial

def build_class_embeddings(all_classes):
    texts = [f"学科领域：{c}" for c in all_classes]
    embs = embed_model.encode(texts, normalize_embeddings=True)
    return {
        c: embs[i] for i, c in enumerate(all_classes)
    }

TRAIN_FILE = "./train.jsonl"
VAL_FILE = "./val.jsonl"
print("正在加载并处理数据集...")
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
all_classes = get_all_classes(TRAIN_FILE)
CLASS_EMB = build_class_embeddings(all_classes)

def select_semantic_rejected(label, all_classes, class_emb, confusable_map=None):
    label_emb = class_emb[label]

    # 1️⃣ 候选集合
    if confusable_map and label in confusable_map:
        candidates = confusable_map[label]
    else:
        candidates = [c for c in all_classes if c != label]

    # 2️⃣ 计算余弦相似度
    sims = {}
    for c in candidates:
        emb = class_emb[c]
        sims[c] = float(np.dot(label_emb, emb))  # 已 normalize

    # 3️⃣ 选最相似的
    rejected = max(sims, key=sims.get)
    return rejected

def process_func_with_dpo(example, all_classes):
    MAX_LENGTH = 512

    # ---------
    # 1. SFT 部分
    # ---------
    messages = example["messages"]
    label = extract_label(example)

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        max_length=MAX_LENGTH,
        truncation=True
    )

    labels = input_ids.copy()

    user_len = len(
        tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            add_generation_prompt=False,
            max_length=MAX_LENGTH,
            truncation=True
        )
    )

    labels[:user_len] = [-100] * user_len

    # ---------
    # 2. DPO 部分
    # ---------
   # negatives = [c for c in all_classes if c != label]
   # neg = random.choice(negatives)
    
    neg = select_semantic_rejected(label=label, all_classes=all_classes, class_emb=CLASS_EMB)


    # prompt（system + user）
    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True
    )

    chosen_text = label
    rejected_text = neg

    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )["input_ids"]

    chosen_ids = tokenizer(
        chosen_text,
        truncation=True,
        max_length=64,
        return_tensors=None
    )["input_ids"]

    rejected_ids = tokenizer(
        rejected_text,
        truncation=True,
        max_length=64,
        return_tensors=None
    )["input_ids"]

    return {
        # ---- SFT ----
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),

        # ---- DPO ----
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }

process_fn = partial(process_func_with_dpo, all_classes=all_classes)
tokenized_dataset = dataset.map(
    process_fn,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing with SFT + DPO"
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

training_args = TrainingArguments(
    output_dir="./dpo_optimize",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    remove_unused_columns=False,
    report_to="none",
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

from torch.nn.utils.rnn import pad_sequence
import torch

def sft_dpo_data_collator(features):
    # ---------
    # SFT 部分
    # ---------
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    # ---------
    # DPO 部分
    # ---------
    prompt_ids = [torch.tensor(f["prompt_ids"], dtype=torch.long) for f in features]
    chosen_ids = [torch.tensor(f["chosen_ids"], dtype=torch.long) for f in features]
    rejected_ids = [torch.tensor(f["rejected_ids"], dtype=torch.long) for f in features]

    prompt_ids = pad_sequence(
        prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    chosen_ids = pad_sequence(
        chosen_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    rejected_ids = pad_sequence(
        rejected_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }


trainer = SFTWithDPOTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=sft_dpo_data_collator,
    dpo_beta=0.1,
    dpo_lambda=0.15,   # 正则权重0.1
    callbacks=[swanlab_callback],
)


trainer.train()


#VAL_FILE = "./val.jsonl"
RESULT_FILE = "./dpo_result.jsonl"
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
