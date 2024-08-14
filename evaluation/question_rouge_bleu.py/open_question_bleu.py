# -*- encoding: utf-8 -*-
'''
Filename         :open_question_bleu.py
Description      :
Time             :2024/04/07 20:45:39
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import json, os
import nltk

from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import jieba
from loguru import logger

# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

def run(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    n_gram = args.n_gram
    if n_gram==1:
        weights = (1, 0, 0, 0)
    elif n_gram==2:
        weights = (0.5, 0.5, 0, 0)
    elif n_gram==3:
        weights = (0.33, 0.33, 0.33, 0)
    elif n_gram==4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        raise ValueError('n_gram must be in [1, 2, 3, 4]')
    with open(data_dir, 'r') as f:
        data = json.load(f)

    predictions = [entry['candidate'] for entry in data]
    references = [entry['output'] for entry in data]

    total_bleu = 0.0
    total_meteor = 0.0
    for pred, ref in zip(predictions, references):
        if args.use_jieba:
            pred_tokens = list(jieba.cut(pred, cut_all=False))
            ref_tokens = list(jieba.cut(ref, cut_all=False))
        else:
            pred_tokens = list(pred.lower())
            ref_tokens = list(ref.lower())
        bleu = sentence_bleu([ref_tokens], pred_tokens, weights=weights)
        meteor = meteor_score([ref_tokens], pred_tokens)
        total_bleu += bleu
        total_meteor += meteor
    average_bleu = total_bleu / len(predictions)
    average_meteor = total_meteor / len(predictions)
    
    logger.info(f"BLEU score: {average_bleu:.3f}, METEOR score: {average_meteor:.3f}")
    
    if save_dir:
        with open(os.path.join(save_dir,), 'w', encoding="utf-8") as f:
            f.write(json.dumps({"bleu": average_bleu, "meteor": average_meteor}, ensure_ascii=False))

    return average_bleu


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="result_data/tests/20240408/results/rc/cmlm-zhongjinggpt.json")
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--n_gram', type=int, default=4)
    parser.add_argument('--use_jieba', type=bool, default=True)
    args = parser.parse_args()
    run(args)

