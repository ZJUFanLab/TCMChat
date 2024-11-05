import json
import re
data_dir = "your/file/path"
save_dir = "your/file/path"

data = json.load(open(data_dir, "r", encoding="utf-8"))

all_data = []
for d in data:
    standard_number = re.findall(r'(\d+(?:\.\d+)?)\s*(?:\(Log10uM\))?', d["output"])
    candidate_number = re.findall(r'(\d+(?:\.\d+)?)\s*(?:\(Log10uM\))?', d["candidate"])
    if len(standard_number) == 0 or len(candidate_number) == 0:
        continue
    standard_number = float(standard_number[0])
    candidate_number = float(candidate_number[0])
    d["standard_number"] = standard_number
    d["candidate_number"] = candidate_number
    all_data.append(d)

json.dump(all_data, 
          open(save_dir, "w", 
               encoding="utf-8"), 
          ensure_ascii=False, 
          indent=4)