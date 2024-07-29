[**中文**](./README_ZH.md) | [**English**](./README.md)

<p align="center" width="100%">
<a href="https://github.com/daiyizheng/TCMChat" target="_blank"><img src="assets/logo.png" alt="TCMChat" style="width: 25%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# TCMChat: Traditional Chinese Medicine Recommendation System based on Large Language Model

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/LICENSE) [![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-390/)

## 新闻

[2024-5-17] huggingface 开源模型权重

## 应用

### 安装

```
git clone https://github.com/daiyizheng/TCMChat
cd TCMChat
```

首先安装依赖包，python环境建议3.10+

```
pip install -r requirements.txt
```

### 权重下载

- [TCMChat](https://huggingface.co/daiyizheng/TCMChat): 基于baichuan2-7B-Chat的中药、方剂知识问答与推荐。

### 推理

#### 命令行测试

```
python cli_infer.py \
--model_name_or_path /your/model/path \
--model_type  chat
```

#### Web页面测试

```
python gradio_demo.py
```

我们提供了一个在线的体验工具：[http://xxx](http://xxx)


### 重新训练

#### 数据集下载

...

#### 预训练

...

#### 微调

...

### 训练细节

请参考论文实验部分说明。

## 引用

```
@misc{tcmchat,
      doi = {xxx},
      url = {xxx},
      author = {Yizheng Dai, Jinlu Zhang, Yulong Chen, Qian Chen, Jie Liao, Fei Chi, Xin Shao, Xiaohui Fan},
      keywords = {TCM, Pre-training, Fine-tuning},
      title = {TCMChat: Traditional Chinese Medicine Recommendation System based on Large Language Model},
      publisher = {arXiv},
      year = {2024},
      copyright = {l}
}
```
