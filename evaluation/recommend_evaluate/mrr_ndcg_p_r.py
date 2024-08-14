# -*- encoding: utf-8 -*-
'''
Filename         :tuijian.py
Description      :
Time             :2024/03/25 14:57:04
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import math
import json

from loguru import logger
import numpy as np

def calculate_mrr(predictions, labels):
    mrr = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]

        if label in prediction:
            rank = prediction.index(label) + 1
            mrr += 1 / rank

    mrr /= len(predictions)
    return mrr


#已经取了前k个
def precision_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list) if len(ground_list) != 0 else 1)
    rec = hits / (1.0 * len(ground_list) if len(ground_list) != 0 else 1)
    return pre, rec
 
def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg
 
def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg if idcg != 0 else 0
 
def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0
 
 
def RR(ranked_list, ground_list):
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)
 
    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)
    #print(hits)
 
    r = np.array(rankedlist)
 
    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)
    #print(float(count / k))
    #print(float(count / len(test_matrix)))
    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)
 
def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)
 
    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)
 
    for c in range(count):
        ap += (c+1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)
 
    if count != 0:
        mrr = 1 / (hits[0][0] + 1)
 
    if count != 0:
        map = ap / count
 
    return map, mrr, float(dcg / idcg)


def HR_K(rankedlist, ground_truth, k=1):
    hit = []
    for i in rankedlist:
        if i in ground_truth:
            return 1
    return 0


def cal_metric(datasets):
    rr = 0
    precision_1 = 0
    recall_1 = 0
    precision_3 = 0
    recall_3 = 0
    hr_1 = 0
    hr_3 = 0
    ndcg_ = 0
    for idx, data in enumerate(datasets):
        candidate_rank = data["candidate_list"]
        rank_list = data["rank_list"]
        ## MRR
        score = RR(candidate_rank, rank_list)
        rr+=score
        ## precision_and_recall k=1
        k = 1
        p, r = precision_and_recall(candidate_rank[:k], rank_list) if candidate_rank[:k] else (0,0)
        precision_1+=p
        recall_1+=r
        ## precision_and_recall k=3
        k=3
        p, r = precision_and_recall(candidate_rank[:k], rank_list) if candidate_rank[:k] else (0,0)
        precision_3+=p
        recall_3+=r
        ## HR
        k=1
        hr = HR_K(candidate_rank[:k], rank_list[:k])
        hr_1+=hr
        ##HR
        k=3
        hr = HR_K(candidate_rank[:k], rank_list[:k])
        hr_3+=hr
        
        ##nDCG
        nd = nDCG(candidate_rank, rank_list)
        ndcg_+=nd
        
        

    logger.info(f"MRR: {rr/len(datasets):.3f}")
    logger.info(f"precision@1:{ precision_1/len(datasets):.3f}")
    logger.info(f"recall@1: {recall_1/len(datasets):.3f}")
    logger.info(f"precision@3: {precision_3/len(datasets):.3f}")
    logger.info(f"recall@3: { recall_3/len(datasets):.3f}")
    logger.info(f"hr@1: { hr_1/len(datasets):.3f}")
    logger.info(f"hr@3: { hr_3/len(datasets):.3f}")
    logger.info(f"nDCG: { ndcg_/len(datasets):.3f}")
    
if __name__ == '__main__':
    data_path = "result_data/tests/20240408/results/changshi/rank_ours.json"
    contents_json = json.load(open(data_path, 'r', encoding='utf-8'))
    cal_metric(contents_json)
    