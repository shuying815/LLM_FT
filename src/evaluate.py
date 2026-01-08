import argparse
import os
import torch
import json
from tqdm import tqdm
from modelscope import AutoTokenizer
from qwen2.modeling_qwen2 import Qwen2ForCausalLM
from qwen2.configuration_qwen2 import Qwen2Config
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PrefixTuningConfig

def main():
    parser = argparse.ArgumentParser(description="测试脚本")
    
    # 添加参数
    parser.add_argument("--ft_method", type=str, default="lora", help="微调方法")
    parser.add_argument("--file_path", type=str, required=True, help="权重文件路径")

    
    # 解析参数
    args = parser.parse_args()
    method = args.ft_method

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 路径配置
    BASE_MODEL_PATH = "./qwen/Qwen2___5-1___5B-Instruct/"
    VAL_FILE = "./val.jsonl"
    RESULT_FILE = "result.jsonl"
    MAX_NEW_TOKENS = 12 

    print(f"正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    if method == 'adapter':
        config = Qwen2Config.from_json_file("./qwen/Qwen2___5-1___5B-Instruct/config.json")
        model = Qwen2ForCausalLM(config)
        model_weights = load_file(args.file_path+'/model.safetensors')
        miss = model.load_state_dict(model_weights, strict=False)
        model = model.to(dtype=torch.bfloat16, device="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, args.file_path)
     
    model.eval()
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
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
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
        return ground_truth in response

    base_correct_count = 0
    ft_correct_count = 0
    total_count = 0
    results_log = []

    print("\n🚀 开始自动化评估...")
    pbar = tqdm(data_samples, desc="Evaluating", unit="sample")

    for sample in pbar:
        total_count += 1

        input_messages = sample["messages"][:-1]
        ground_truth = sample["messages"][-1]["content"]

        ft_response = predict(input_messages, model, tokenizer)
        print(ft_response)
        is_ft_correct = check_correctness(ft_response, ground_truth)
        if is_ft_correct:
            ft_correct_count += 1

        results_log.append({
            "input": input_messages[-1]["content"], # 记录最后一个问题
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

if __name__ == "__main__":
    main()