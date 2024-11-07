[**中文**](./README_ZH.md) | [**English**](./README.md)

<p align="center" width="100%">
<a href="https://github.com/daiyizheng/TCMChat" target="_blank"><img src="assets/logo.png" alt="TCMChat" style="width: 25%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# TCMChat: A Generative Large Language Model for Traditional Chinese Medicine

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/blob/main/LICENSE) [![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-390/)

## News
[*November 1, 2024*] We have fully open-sourced the model weights and training dataset on Huggingface.                   
[*May 17, 2024*] Open source model weight on HuggingFace.                 

## Application

### Install
```shell
git clone https://github.com/ZJUFanLab/TCMChat
cd TCMChat
```
Create a conda environment
```shell
conda create -n baichuan2 python=3.10 -y
```
First install the dependency package. python environment 3.10+ is recommended.

```shell
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

- [Pretrain dataset](https://huggingface.co/datasets/ZJUFanLab/TCMChat-dataset-600k) 
- [SFT dataset](https://huggingface.co/datasets/ZJUFanLab/TCMChat-dataset-600k)
- [Benchmark dataset](https://github.com/ZJUFanLab/TCMChat/tree/master/evaluation/resources)


> Note: Before performing pre-training, fine-tuning, and inference, please modify the relevant paths for your model, data, and other related files.
#### Pre-training

```shell
## Slurm cluster
sbatch scripts/pretrain/baichuan2_7b_chat.slurm
## or
bash scripts/pretrain/baichuan2_7b_chat.sh
```

#### Fine-tuning
```shell
## Slurm cluster
sbatch scripts/sft/baichuan2_7b_chat.slurm
## or
bash scripts/sft/baichuan2_7b_chat.sh
```
### Training details

Please refer to the experimental section of the paper for instructions.


### Benchmark evaluation

#### Choice Question
```shell
python evaluation/choices_evaluate/eval.py   --model_path_or_name /your/model/path --model_name  baichuan2-7b-chat --few_shot -sz herb --dev_file_path evaluation/resources/choice/single/tcm-herb_dev.csv --val_file_path evaluation/resources/choice/single/choice_herb_500.csv --log_dir logs/choices
```

#### Reading Comprehension
```shell
python infers/baichuan_infer.py \
--model_name_or_path /your/model/path / \
--model_type chat \
--save_path /your/save/data/path \
--data_path /your/data/path
##BertScore
python evaluation/question_rouge_bleu.py/question_bert_score.py
## BLEU METEOR
python evaluation/question_rouge_bleu.py/open_question_bleu.py
## ROUGE-x
python evaluation/question_rouge_bleu.py/open_question_rouge.py

```

#### Entity Extraction
```shell
python infers/baichuan_infer.py \
--model_name_or_path /your/model/path / \
--model_type chat \
--save_path /your/save/data/path \
--data_path /your/data/path

python evaluation/ner_evaluate/tcm_entity_recognition.py

```

#### Medical Case Diagnosis
```shell
python infers/baichuan_infer.py \
--model_name_or_path /your/model/path / \
--model_type chat \
--save_path /your/save/data/path \
--data_path /your/data/path

python evaluation/acc_evaluate/extract_syndrome.py

```

#### Herb or Formula Recommendation
```shell
python infers/baichuan_infer.py \
--model_name_or_path /your/model/path / \
--model_type chat \
--save_path /your/save/data/path \
--data_path /your/data/path

python evaluation/recommend_evaluate/mrr_ndcg_p_r.py

```
### ADMET Prediction
#### Regression
```shell
python infers/baichuan_infer.py \
--model_name_or_path /your/model/path / \
--model_type chat \
--save_path /your/save/data/path \
--data_path /your/data/path

python evaluation/admet_evaluate/rmse_mae_mse.py

```
#### Classification
```shell
python infers/baichuan_infer.py \
--model_name_or_path /your/model/path / \
--model_type chat \
--save_path /your/save/data/path \
--data_path /your/data/path

python evaluation/admet_evaluate/acc_recall_f1.py

```