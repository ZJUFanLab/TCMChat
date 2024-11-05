[**中文**](./README_ZH.md) | [**English**](./README.md)

<p align="center" width="100%">
<a href="https://github.com/daiyizheng/TCMChat" target="_blank"><img src="./logo.png" alt="TCMChat" style="width: 25%; min-width: 300px; display: block; margin: auto;"></a>
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

- [Pretrain dataset](https://github.com/ZJUFanLab/TCMChat/tree/master/data/pretrain) 
- [SFT dataset](https://github.com/ZJUFanLab/TCMChat/tree/master/data/sft)
- [Benchmark dataset](https://github.com/ZJUFanLab/TCMChat/tree/master/data/evaluate)

> Note: Currently only sample data is provided. In the near future, we will fully open source the original data.


#### Pre-training

```shell
train_type="pretrain"
train_file="data/pretrain/train"
validation_file="data/pretrain/test"
block_size="1024"
deepspeed_dir="data/resources/deepspeed_zero_stage2_config.yml"
num_train_epochs="2"
export WANDB_PROJECT="TCM-${train_type}"
date_time=$(date +"%Y%m%d%H%M%S")
run_name="${date_time}_${block_size}"
model_name_or_path="your/path/Baichuan2-7B-Chat"
output_dir="output/${train_type}/${date_time}_${block_size}"


accelerate launch --config_file ${deepspeed_dir} src/pretraining.py \
--model_name_or_path ${model_name_or_path}  \
--train_file  ${train_file}  \
--validation_file ${validation_file}  \
--preprocessing_num_workers 20  \
--cache_dir ./cache \
--block_size  ${block_size}  \
--seed 42  \
--do_train  \
--do_eval  \
--per_device_train_batch_size  32  \
--per_device_eval_batch_size  32  \
--num_train_epochs ${num_train_epochs}  \
--low_cpu_mem_usage  True \
--torch_dtype bfloat16  \
--bf16  \
--ddp_find_unused_parameters False  \
--gradient_checkpointing True  \
--learning_rate 2e-4 \
--warmup_ratio 0.05 \
--weight_decay 0.01 \
--report_to wandb  \
--run_name ${run_name}  \
--logging_dir  logs \
--logging_strategy steps \
--logging_steps 10 \
--eval_steps 50 \
--evaluation_strategy steps \
--save_steps 100 \
--save_strategy steps \
--save_total_limit 13 \
--output_dir  ${output_dir}  \
--overwrite_output_dir
```

#### Fine-tuning

```shell
train_type="SFT"
model_max_length="1024"
date_time=$(date +"%Y%m%d%H%M%S")
data_path="data/sft/sample_train_baichuan_data.json"
model_name_or_path="your/path/pretrain"
deepspeed_dir="data/resources/deepspeed_zero_stage2_confi_baichuan2.json"
export WANDB_PROJECT="TCM-${train_type}"
run_name="${train_type}_${date_time}"
output_dir="output/${train_type}/${date_time}_${model_max_length}"


deepspeed --hostfile="" src/fine-tune.py  \
    --report_to "wandb" \
    --run_name ${run_name}  \
    --data_path ${data_path} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${output_dir} \
    --model_max_length ${model_max_length} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ${deepspeed_dir} \
    --bf16 True \
    --tf32 True
```

### Training details

Please refer to the experimental section of the paper for instructions.




