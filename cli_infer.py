# -*- encoding: utf-8 -*-
'''
Filename         :cli_infer.py
Description      :
Time             :2024/05/16 17:33:07
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from typing import Any
import logging
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


logger = logging.getLogger(__name__)

class Baichuan2(object):
    def __init__(self, 
                 model_name_or_path:str, 
                 model_type:str="chat") -> None:
        self.model_type = model_type
        logger.info("--------------> 初始化模型")
        if model_type == "chat":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                                           use_fast=False, 
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                              device_map="auto", 
                                                              torch_dtype=torch.bfloat16, 
                                                              trust_remote_code=True)
            self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                              device_map="auto", 
                                                              trust_remote_code=True)
        logger.info("--------------> 初始化模型结束")
        
    def __call__(self, message:str) -> Any:
        if self.model_type=="chat":
            messages = []
            messages.append({"role": "user", "content": message})
            response = self.model.chat(self.tokenizer, messages)
        else:
            inputs = self.tokenizer(message, return_tensors='pt')
            inputs = inputs.to('cuda:0')
            pred = self.model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
            response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)          
        return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,default="/slurm/home/yrd/shaolab/daiyizheng/projects/tcm/output/SFT/baichuan2_7b_chat_1024_epoch_4_20241026_1810605_20241028092615_1024", help='model_name_or_path')
    parser.add_argument('--model_type', type=str, default="chat", help='model_type')
    
    args = parser.parse_args()
    
    model_name_or_path = args.model_name_or_path
    model_type = args.model_type

    baichuan = Baichuan2(model_name_or_path=model_name_or_path, 
                         model_type=model_type)
    while True:
        message = input("input:")
        if message.strip() == "exit":
            break
        res = baichuan(message=message)
        print(res)
        for r in res:
            print(r)
        
"""
python cli_baichuan2.py \
--model_name_or_path xxx \
--model_type  chat
"""