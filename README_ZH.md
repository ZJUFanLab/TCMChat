[**中文**](./README_ZH.md) | [**English**](./README.md)

<p align="center" width="100%">
<a href="https://github.com/daiyizheng/TCMChat" target="_blank"><img src="./logo.png" alt="TCMChat" style="width: 25%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# TCMChat: Traditional Chinese Medicine Recommendation System based on Large Language Model

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/LICENSE) [![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-390/)

## 新闻
[2024-11-1] 我们在Huggingface上完全开源了模型权重和训练数据集
[2024-5-17] huggingface 开源模型权重


## 应用

### 安装
```shell
git clone https://github.com/daiyizheng/TCMChat
cd TCMChat
```

创建conda 环境
```shell
conda create -n baichuan2 python=3.10 -y
```

首先安装依赖包，python环境建议3.10+
``` shell
pip install -r requirements.txt
```

### 权重下载
- [TCMChat](https://huggingface.co/daiyizheng/TCMChat): 基于baichuan2-7B-Chat的中药、方剂知识问答与推荐。

### 推理

#### 命令行测试

```shell
python cli_infer.py \
--model_name_or_path /your/model/path \
--model_type  chat
```

#### Web页面测试

```shell
python gradio_demo.py
```
我们提供了一个在线的体验工具：[https://xomics.com.cn/tcmchat](https://xomics.com.cn/tcmchat)


### 重新训练
#### 数据集下载

- [预训练数据](https://github.com/ZJUFanLab/TCMChat/tree/master/data/pretrain) 
- [微调数据](https://github.com/ZJUFanLab/TCMChat/tree/master/data/sft)
- [基准评测数据](https://github.com/ZJUFanLab/TCMChat/tree/master/data/evaluate)


> 注意： 在执行预训练、微调和推理之前，请修改自己模型、数据等相关数据路径
#### 预训练

```shell
## slurm 集群
sbatch scripts/pretrain/baichuan2_7b_chat.slurm
##或者
bash scripts/pretrain/baichuan2_7b_chat.sh
```

#### 微调
```shell
## slurm 集群
sbatch scripts/sft/baichuan2_7b_chat.slurm
##或者
bash scripts/sft/baichuan2_7b_chat.sh
```

### 基准评估


### 训练细节

请参考论文实验部分说明。

