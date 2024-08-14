# -*- encoding: utf-8 -*-
'''
Filename         :llama.py
Description      :
Time             :2024/01/25 11:04:20
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from loguru import logger

from evaluator import Evaluator


class LLaMA_HuggingFace_Evaluator(Evaluator):
    def __init__(self, 
                 choices, 
                 model_name,
                 model_path_or_name, 
                 k=-1) -> None:
        super().__init__(choices, model_name, k)
        logger.info("************Model initialization start*****************")
        self.model = LlamaForCausalLM.from_pretrained(
            model_path_or_name, 
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported() 
                else torch.float32
                ))
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_path_or_name, 
            trust_remote_code=True
            )
        self.model_path_or_name = model_path_or_name
        self.patterns = [
            "答案是?\s?([ABCD])",
            "答案是?\s?：([ABCD])",
            "答案是?\s?:([ABCD])",
            "答案应该?是\s?([ABCD])",
            "答案应该?选\s?([ABCD])",
            "答案为\s?([ABCD])",
            "选择\s?([ABCD])",
            "只有选?项?\s?([ABCD])\s?是?对",
            "只有选?项?\s?([ABCD])\s?是?错",
            "只有选?项?\s?([ABCD])\s?不?正确",
            "只有选?项?\s?([ABCD])\s?错误",
            "说法不?对选?项?的?是\s?([ABCD])",
            "说法不?正确选?项?的?是\s?([ABCD])",
            "说法错误选?项?的?是\s?([ABCD])",
            "([ABCD])\s?是正确的",
            "([ABCD])\s?是正确答案",
            "选项\s?([ABCD])\s?正确",
            "所以答\s?([ABCD])",
            "1.\s?([ABCD])[.。$]?$",
            "所以\s?([ABCD][.。$]?$)",
            "所有\s?([ABCD][.。$]?$)",
            "[\s，：:,]([ABCD])[。，,\.]?$",
            "[\s，,：:][故即]([ABCD])[。\.]?$",
            "[\s，,：:]因此([ABCD])[。\.]?$",
            "[是为。]\s?([ABCD])[。\.]?$",
            "因此\s?([ABCD])[。\.]?$",
            "显然\s?([ABCD])[。\.]?$",
            "1.\s?(.*?)$",
            "答案是\s?(\S+)(?:。|$)",
            "答案应该是\s?(\S+)(?:。|$)",
            "答案为\s?(\S+)(?:。|$)",
            ]
        logger.info("************Model initialization end*****************")
        
    def eval_subject(self, 
                     subject_zh, 
                     test_df, 
                     choices_type_name,
                     dev_df=None, 
                     few_shot=False, 
                     save_result_dir=None,
                     cot=False):
        result = []
        score = []
        few_shot_prompt = self.generate_few_shot_prompt(
                                        subject_zh, 
                                        dev_df, 
                                        choices_type_name,
                                        cot=cot) if few_shot else ""
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            full_prompt = few_shot_prompt + question
            pred = self.inference(full_prompt, return_logits= not cot)
            correct = 1 if pred == row['answer'] else 0
            result.append(pred)
            score.append(correct)
        correct_ratio = 100*sum(score)/len(score)
        if save_result_dir:
            test_df['model_output'] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(
                save_result_dir, f'{subject_zh}_test.csv'), encoding="utf-8", index=False)
        return correct_ratio
    
    @torch.inference_mode()        
    def inference(self, prompt, return_logits: bool = False):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()

        logits = self.model(input_ids=input_ids).logits[:,-1].flatten()

        candidate_logits = [logits[self.tokenizer(label).input_ids[-1]] for label in self.choices]
        candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
        probs = (torch.nn.functional.softmax(candidate_logits,dim=0).detach().cpu().numpy())
        answer = {i: k for i, k in enumerate(self.choices)}[np.argmax(probs)]
        return answer

    def generate_few_shot_prompt(self, 
                                 subject, 
                                 dev_df, 
                                 choices_type_name, 
                                 cot=False):
        
        prompt = f"以下是中医药关于{subject}考试的{choices_type_name}，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot
            )
        return prompt
    
    def format_example(self, line, include_answer=True, cot=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + \
                    line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += '\n答案：' + line["answer"] + '\n\n'
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += '\n答案：'
        return example