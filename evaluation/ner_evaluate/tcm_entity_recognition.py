# -*- encoding: utf-8 -*-
'''
Filename         :tcm_entity.py
Description      :
Time             :2024/03/19 09:16:40
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import json
import argparse
from loguru import logger

def calculate_f1_score(true_entities, predicted_entities):
    true_entities = set(true_entities)
    predicted_entities = set(predicted_entities)
    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if len(predicted_entities) > 0 else 0
    recall = true_positive / len(true_entities) if len(true_entities) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # print(precision,recall,f1_score)
    return precision,recall,f1_score

def calculate_accuracy(question_data):
    correct_count_p = 0
    correct_count_r = 0
    correct_count_f = 0
    total_count = len(question_data)
    for i, question in enumerate(question_data):
        true_output = [i.split(":")[-1] for i in question['output'].split("；")]
        true_outputs = []
        for tt in true_output:
            true_outputs.extend([i.strip() for i in tt.split("，") if i.strip()!=""])
        my_output = [i.split(":")[-1] for i in question['candidate'].split("；")]
        my_outputs = []
        for tt in my_output:
            my_outputs.extend([i.strip() for i in tt.split("，") if i.strip()!=""])
        
        precision,recall,f1_score = calculate_f1_score(true_outputs, my_outputs)
        # print(f1_score)
        correct_count_p+=precision
        correct_count_r+=recall
        correct_count_f+=f1_score
        
    return correct_count_p/total_count,correct_count_r/total_count,correct_count_f/total_count

def run(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    with open(data_dir, "r") as json_file:
        json_data = json.load(json_file)
        p, r, f = calculate_accuracy(json_data)
    logger.info(f"precision: {p:.3f}，recall: {r:.3f}，f1_score: {f:.3f}")
    if save_dir:
        with open(save_dir, "w") as json_file:
            json.dump({"precision": p, "recall": r, "f1_score": f}, json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="result_data/tests/20240408/results/ner/extract_bentsao_literature.json")
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    run(args)