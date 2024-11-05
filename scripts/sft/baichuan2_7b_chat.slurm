#!/bin/bash

#SBATCH --job-name=your_job_name
#SBATCH --partition=xx
#SBATCH --ntasks=xx
#SBATCH --gres=gpu:xx
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --mail-type=ALL         
#SBATCH --mail-user=xxx@xxx.com

conda activate baichuan2


train_type="SFT"
model_max_length="1024"
date_time=$(date +"%Y%m%d%H%M%S")
data_path="your/sft/train_data"
model_name_or_path="your/output/pretrain"
deepspeed_dir="resources/deepspeed_zero_stage2_config_baichuan2.json"
export WANDB_PROJECT="TCM-${train_type}"
export WANDB_MODE="offline"
run_name="your_experiment_name"
output_dir="output/${train_type}/${run_name}"


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
