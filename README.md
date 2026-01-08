# LLM_FT

### 环境配置

通过下面的命令搭建conda环境完成安装配置。

```bash
conda create -n qwen python=3.9
conda activate qwen
pip install -r requirements.txt
```

### 实验数据集

数据集可以从[zh\_cls\_fudan-news](https://www.modelscope.cn/datasets/swift/zh_cls_fudan-news)下载，或者参考`src`文件夹下的`train.jsonl`和`val.jsonl`文件。

### 目录结构

下面展示了仓库代码文件结构。其中，所有微调相关的代码和数据文件都存放在`src`文件夹下。`results`文件夹下存储了本次实验的所有实验结果和对应的模型权重，其中，`ablation`存储了消融实验结果，`accuracy`存储了所有微调方法的测试结果，`hyperparameter`存储了超参实验结果。运行命令行文件在`scripts`文件夹下，所有训练好的模型权重文件存放于`ckpt`文件夹下。`requirements.txt`文件为环境配置文件。

```bash
LLM_FT/
|--ckpt/ 
|    |--lora/
|    |--prefix/
|    |--adapter/
|    |--dpo/
|    |--dpo_optimize/
|--src/ 
|    |--train.jsonl
|    |--val.jsonl
|    |--lora.py
|    |--prefix.py
|    |--adapter.py
|    |--dpo.py
|    |--dpo_optimize.py
|--results/
|    |--ablation
|    |--accuracy
|    |--hyperparameter
|--scripts/
|    |--run.sh
|--requirements.txt
|--README.md
```

### 训练

如果需要执行某个微调方法，更改命令行参数即可，可选参数为：lora、prefix、adapter、dpo以及dpo\_optimize。下面是运行LoRA微调的示例。

```bash
bash scripts/run.sh lora
```

### 测试

微调完成后模型会自动在验证集上执行测试，如果需要使用微调好的权重直接执行测试，命令如下，将`checkpoint\_path`更换成权重文件路径即可。测试会输出在数据集上的预测准确的样本数量和准确率，同时会生成每条数据模型对应的输出构成的文件`result.jsonl`。

```bash
bash scripts/test.sh lora checkpoint_path
```

#### 实验结果

训练曲线可视化。

<img src=".\results\train.png" alt="contrast_result" width="500" height="749" />

微调实验结果对比（灰色表示基于Qwen2.5-7B模型的微调，其余表示基于Qwen2.5-1.5B模型的微调）。

<img src=".\results\accuracy\contrast_result.png" alt="contrast_result" style="zoom:50%;" />

DPO优化方法消融实验结果。

<img src=".\results\ablation\ablation.png" alt="ablation" style="zoom:50%;" />

Adaptet 微调 reduction_dim超参实验。


<img src=".\results\hyperparameter\reduction_dim.png" alt="reduction_dim" style="zoom:50%;" />


