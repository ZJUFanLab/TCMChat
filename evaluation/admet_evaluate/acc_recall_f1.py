# -*- encoding: utf-8 -*-
'''
Filename         :acc_recall_f1.py
Description      :
Time             :2024/04/15 16:10:29
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import json

import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



data_dir = "your/file/path"
if data_dir.endswith('.json'):
    data = json.load(open(data_dir, 'r', encoding='utf-8'))
    standard_list = []
    candidate_list = []
    for d in data:
        standard_list.append(d['standard_cls'])
        candidate_list.append(d['candidate_cls'])
elif data_dir.endswith('.csv'):
    df = pd.read_csv(data_dir)
    standard_list = df['standard_cls'].tolist()
    candidate_list = df['candidate_cls'].tolist()

acc = accuracy_score(standard_list, candidate_list)
roc_auc = roc_auc_score(standard_list, candidate_list)
print(f"roc-auc: {roc_auc:.2f}")
precision = precision_score(standard_list, candidate_list, average='macro')
recall = recall_score(standard_list, candidate_list, average='macro')
f1 = f1_score(standard_list, candidate_list, average='macro')
logger.info(f"acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
