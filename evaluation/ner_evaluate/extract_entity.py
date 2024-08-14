# -*- encoding: utf-8 -*-
'''
Filename         :extract_entity.py
Description      :
Time             :2024/04/08 14:37:26
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import json

data_dir = "result_data/tests/20240408/results/ner/gemini-pro.json"
save_dir = "result_data/tests/20240408/results/ner/extract_gemini-pro.json"
data = json.load(open(data_dir, "r", encoding="utf-8"))

all_data = []
for d in data:
    candidate = d["candidate"]
    d["candidate_ori"] = candidate
    output = d["output"]
    output_list = output.split("；")
    output_list = [i for i in output_list if i.strip() != ""]
    new_vs_list = []
    for o in output_list:
        o_list = o.split(":")
        k = o_list[0]
        vs_list = o_list[1].split("，")
        new_v_list = []
        for v in vs_list:
            if v in candidate:
                new_v_list.append(v)
        new_vs_list.append((k, new_v_list))
    new_candidate = ""
    for nv in new_vs_list:
        if len(nv[1]) == 0:
            continue
        new_candidate += nv[0] + ":" + "，".join(nv[1]) + "；"
    d["candidate"] = new_candidate
    all_data.append(d)

json.dump(all_data, 
          open(save_dir, "w", 
               encoding="utf-8"), 
          ensure_ascii=False, 
          indent=4)
        
                
        
