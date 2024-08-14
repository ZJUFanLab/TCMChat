# -*- encoding: utf-8 -*-
'''
Filename         :提取生成中药-方剂信息.py
Description      :
Time             :2024/03/25 13:35:08
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import json
from copy import deepcopy

import pandas as pd

def build_index(list, text):
    data = []
    for l in list:
        idx = text.index(l)
        data.append((idx, l))
    data = sorted(data, key=lambda x: x[0])
    return [x[1] for x in data]

def remove_dump(list):
    ## 去除重复
    list = sorted(list, key=lambda x: len(x), reverse=True)
    arr = []
    for i in list:
        if len(arr)==0:
            arr.append(i)
            continue
        else:
            arr_copy = deepcopy(arr)
            flag = True
            for j in arr_copy:
                if i in j or j in i:
                    flag = False
                    break
            if flag:
                arr.append(i)
    return arr

def build_dictionary(data_dir):
    with open(data_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines if i.strip() != ""]
        line_list = []
        for line in lines:
            words = line.split("|||")[-1]
            line_list.extend(words.split("、"))
        return list(set(line_list))
            

herb_dir = "result_data/字典/中药/all-功效-中药(have etcm).txt"
formula_dir = "result_data/字典/方剂/all_功效-方剂(have etcm).txt"

herb_list = build_dictionary(herb_dir)
formula_list = build_dictionary(formula_dir)
herb_list = [i for i in herb_list if not pd.isna(i)]
formula_list = [i for i in formula_list if not pd.isna(i)]

## 
data_path = "result_data/tests/20240408/results/changshi/bentsao_med.json"
save_path = "result_data/tests/20240408/results/changshi/rank_bentsao_med.json"

contents_json = json.load(open(data_path, "r", encoding="utf-8"))
all_contents = []
for idx, data in enumerate(contents_json):
    if "rank" not in data:
        print(data)
        raise ValueError 
    candidate = data["candidate"]
    tishi = data["tishi"]
    rank = data["rank"]
    rank_list = []
    if "herb" in rank or "formula" in rank:
        rank_list.extend(rank["herb"])
        rank_list.extend(rank["formula"])
    else:
        rank_list = rank
        
    candidate_rank = []
    
    for h in herb_list:
        if h in candidate:
            candidate_rank.append(h)
    candidate_rank = remove_dump(candidate_rank)
    
    for h in formula_list:
        if h in candidate:
            candidate_rank.append(h)
    candidate_rank = remove_dump(candidate_rank)
    
    candidate_rank = build_index(candidate_rank, candidate)
    data["candidate_list"] = candidate_rank
    data["rank_list"] = rank_list
    
    all_contents.append(data)

json.dump(all_contents, 
          open(save_path, "w", encoding="utf-8"),
          ensure_ascii=False,
          indent=2)
