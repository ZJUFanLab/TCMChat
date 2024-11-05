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

train_type="pretrain"
train_file="your/pretrain/train_data"
validation_file="your/pretrain/test_data"
block_size="1024"
deepspeed_dir="resources/deepspeed_zero_stage2_config.yml"
num_train_epochs="2"
export WANDB_PROJECT="TCM-${train_type}"
export WANDB_MODE="offline"
date_time=$(date +"%Y%m%d%H%M%S")
run_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${date_time}_${block_size}"
model_name_or_path="/slurm/home/yrd/shaolab/daiyizheng/resources/hf_weights/baichuan/Baichuan2-7B-Chat"
output_dir="output/${train_type}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${date_time}_${block_size}"


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