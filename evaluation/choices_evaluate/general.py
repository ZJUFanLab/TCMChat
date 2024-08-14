# -*- encoding: utf-8 -*-
'''
Filename         :general.py
Description      :
Time             :2024/03/21 11:30:16
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import os
import re

from tqdm import tqdm
import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from evaluator import Evaluator


class General_Evaluator(Evaluator):
    def __init__(self, choices, model_path_or_name, model_name,  k, is_half=False):
        super().__init__(choices, model_name, k)
        logger.info("************Model initialization start*****************")
        self.model = AutoModel.from_pretrained(
        model_path_or_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_bf16_supported() 
            else torch.float32
            )
        )
        self.model = self.model if not is_half else self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_name, 
            trust_remote_code=True
            )
        
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
        
        self.model_path_or_name = model_path_or_name
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
           
            pred = self.generate(full_prompt, with_logits= not cot)
            if not cot:
                correct = 1 if pred == row['answer'] else 0
            else:
                pred, correct = self.extract_answer(row, pred)
            result.append(pred)
            score.append(correct)

        correct_ratio = 100*sum(score)/len(score)
        if save_result_dir:
            test_df['model_output'] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(
                save_result_dir, f'{subject_zh}_test.csv'), encoding="utf-8", index=False)
        return correct_ratio
    
    def extract_answer(self, row, output):
        pred = {"A":0, "B":1, "C":2, "D":3}
        correct_answer_str=self.answer_str(row['answer'], row['A'],row['B'],row['C'],row['D'])
        generate_answer=self.extract_model_answer(str(output), row['A'],row['B'],row['C'],row['D'])
        model_answer=self.extract_answer_option(generate_answer)
        if row['answer']==model_answer or correct_answer_str==model_answer:
            return pred[model_answer] if model_answer in pred else model_answer, 1
        else:
            return pred[model_answer] if model_answer in pred else model_answer, 0
        
    def extract_model_answer(self,text, a,b,c,d):
        option_str=re.escape('A. '+a+'\nB. '+b+'\nC. '+c+'\nD. '+d)
        match = re.search(rf'{option_str}([\s\S]*)$', text)
        if match:
            return match.group(1)
        else:
            return None
    
    def extract_answer_option(self,text):
        match = re.findall(r'(让我们一步一步思考[\s\S]+?)(?:(?=让我们一步一步思考)|$)', text)
        text=match[0]
        regexes = [re.compile(pattern) for pattern in self.patterns]
        for regex in regexes:
            match = regex.search(text)
            if match:
                return match.group(1)
        return None
    
    def answer_str(self,answer,a,b,c,d):
        if answer=='D':
            return d
        elif answer=='C':
            return c
        elif answer=='B':
            return b
        else:
            return a
    
    @torch.inference_mode()           
    def generate(self, 
                  prompt, 
                  with_logits: bool = False):

        if with_logits:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            logits = self.model(input_ids=input_ids).logits[:,-1].flatten()
            candidate_logits = [logits[self.tokenizer(label).input_ids[-1]] for label in self.choices]
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (torch.nn.functional.softmax(candidate_logits,dim=0).detach().cpu().numpy())
            answer = {i: k for i, k in enumerate(self.choices)}[np.argmax(probs)]
            return answer
        
        if self.model_path_or_name.endswith("Chat"):
            messages = []
            messages.append({"role": "user", "content": prompt})
            return self.model.chat(self.tokenizer, messages)
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs = inputs.cuda()
            pred = self.model.generate(**inputs, 
                                       max_new_tokens=64, 
                                       repetition_penalty=1.1)
            return self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        
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
    
    def extract_cot_answer(self):
        pass
    
    
    def generate_dist(self):
        pass