# -*- encoding: utf-8 -*-
'''
Filename         :extract_syndrome.py
Description      :
Time             :2024/04/15 15:47:47
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import json


syndrome_list = []
syndrome_dir = "raw_data/阿里天池/TCM_SD_train_dev/all_syndrome_treaments.txt"
data_dir = "result_data/tests/20240408/results/yian/bentsao_literature.json"
save_dir = "result_data/tests/20240408/results/yian/syndrome_bentsao_literature.jsonn"
with open(syndrome_dir, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line_list = line.split("|||")
        syndrome_list.append(line_list[0])

 
data = json.load(open(data_dir, "r", encoding="utf-8"))
all_data = []
for d in data:
    output = d["output"]
    candidate = d["candidate"]
    syndrome_standard = ""
    for s in syndrome_list:
        if  s in output:
            syndrome_standard = s
            break
    syndrome_output = ""
    for c in syndrome_list:
        if c in candidate:
            syndrome_output = c
            break
    d["syndrome_standard"] = syndrome_standard
    d["syndrome_output"] = syndrome_output
    all_data.append(d)
    
json.dump(all_data, open(save_dir, "w", encoding="utf-8"), ensure_ascii=False)
        
    
    

