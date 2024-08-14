# -*- encoding: utf-8 -*-
'''
Filename         :open_question_rouge.py
Description      : 
Time             :2024/03/18 15:21:36
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
code from https://github.com/zjunlp/Mol-Instructions/blob/main/evaluation/biotext/open_question_rouge.py
'''
import argparse, json

from rouge import Rouge
from loguru import logger
import jieba
import sys
sys.setrecursionlimit(100000) #例如这里设置为十万 


def open_question_rouge(predictions, references):
    rouge = Rouge()
    scores = rouge.get_scores(hyps=predictions, 
                              refs=references, avg=True)
    return scores

def run(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    with open(data_dir, 'r') as f:
        data = json.load(f)
    predictions = [entry['candidate'] for entry in data]
    references = [entry['output'] for entry in data]
    new_predictions = []
    new_references = []
    for pred, ref in zip(predictions, references):
        if args.use_jieba:
            pred_tokens = list(jieba.cut(pred, cut_all=False))
            ref_tokens = list(jieba.cut(ref, cut_all=False))
        else:
            pred_tokens = list(pred.lower())
            ref_tokens = list(ref.lower())
        if len(pred_tokens)<1:
            continue
        
        new_predictions.append(' '.join(pred_tokens))
        new_references.append(' '.join(ref_tokens))

    scores = open_question_rouge(new_predictions, new_references)
    logger.info(f"Open-QA ROUGE scores: {scores}")
    if save_dir:
        with open(save_dir, 'w') as f:
            json.dump(scores, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="result_data/tests/20240408/results/rc/cmlm-zhongjinggpt.json")
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--use_jieba', type=bool, default=False)
    args = parser.parse_args()
    run(args)