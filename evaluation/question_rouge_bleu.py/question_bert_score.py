# -*- encoding: utf-8 -*-
'''
Filename         :question_bert_score.py
Description      :
Time             :2024/04/07 22:44:58
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import json, argparse
from loguru import logger
from bert_score import score


def run(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    with open(data_dir, 'r') as f:
        data = json.load(f)

    predictions = [entry['candidate'] for entry in data]
    references = [entry['output'] for entry in data]

    P, R, F1 = score(predictions, references, model_type="bert-base-chinese", lang="en", verbose=True)
    logger.info(f"p:{P.mean():.3f}, r:{R.mean():.3f}, f1-score: {sum(F1)/len(F1):.3f}")
    if save_dir:
        with open(save_dir, 'w') as f:
            json.dump({'P': P.mean(), 'R': R.mean(), 'F1': sum(F1)/len(F1)}, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="result_data/tests/20240408/results/rc/cmlm-zhongjinggpt.json")
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    run(args)
    
    