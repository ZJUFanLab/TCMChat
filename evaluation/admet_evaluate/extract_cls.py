# Description
# 
# Date: 2024-10-28
# Author: extract_cls.py
# Version: 1.0
# License: extract_cls.py.py
# 
# CHANGELOG:
# Date | Version | Author | Description
# -----| --------| ------ | -----------
# YYYY-MM-DD | 1.0.0 | extract_cls.py | Initial creation

import json

data_dir = "your/file/path"
save_dir = "your/file/path"
data = json.load(open(data_dir, "r", encoding="utf-8"))

all_data = []
for d in data:
    standard_cls = 1 if d["output"]=="有" else 0
    candidate_cls = 1 if d["candidate"]=="有" else 0
    d["standard_cls"] = standard_cls
    d["candidate_cls"] = candidate_cls
    all_data.append(d)

json.dump(all_data, 
          open(save_dir, "w", encoding="utf-8"), 
          ensure_ascii=False,
          indent=2)

