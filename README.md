[**中文**](./README_ZH.md) | [**English**](./README.md)

<p align="center" width="100%">
<a href="https://github.com/daiyizheng/TCMChat" target="_blank"><img src="assets/logo.png" alt="TCMChat" style="width: 25%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# TCMChat: A Generative Large Language Model for Traditional Chinese Medicine

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/LICENSE) [![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-390/)

## News

[2024-5-17] Open source model weight on HuggingFace.                 

## Application

### Install

```
git clone https://github.com/daiyizheng/TCMChat
cd TCMChat
```
First install the dependency package. python environment 3.10+ is recommended.

```
pip install -r requirements.txt
```

### Weights download

- [TCMChat](https://huggingface.co/daiyizheng/TCMChat): QA and recommendation of TCM knowledge based on baichuan2-7B-Chat.

### Inference

#### Command line

```
python cli_infer.py \
--model_name_or_path /your/model/path \
--model_type  chat
```

#### Web demo

```
python gradio_demo.py
```

We provide an online tool：[https://xomics.com.cn/tcmchat](https://xomics.com.cn/tcmchat)


### Retrain

#### Dataset Download

...

#### Pre-training

...

#### Fine-tuning

...

### Training details

Please refer to the experimental section of the paper for instructions.

## Citing 

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
