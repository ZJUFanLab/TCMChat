# Description
# 
# Date: 2024-10-27
# Author: rmse_mae_mse.py
# Version: 1.0
# License: rmse_mae_mse.py.py
# 
# CHANGELOG:
# Date | Version | Author | Description
# -----| --------| ------ | -----------
# YYYY-MM-DD | 1.0.0 | rmse_mae_mse.py | Initial creation

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
import json

data_dir = "your/file/path"

if data_dir.endswith(".json"):
    data = json.load(open(data_dir, "r", encoding="utf-8"))
    y_test = [float(d["standard_number"]) for d in data]
    y_pred = [float(d["candidate_number"]) for d in data]

elif data_dir.endswith(".csv"):
    df = pd.read_csv(data_dir)
    y_test = df["standard_number"]
    y_pred = df["candidate_number"]

# 计算MSE和RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)

print(f"均方误差: {mse:.2f}")
print(f"均方根误差: {rmse:.2f}")
print(f"平均绝对误差: {mae:.2f}")

# 计算R-squared
r2 = r2_score(y_test, y_pred)
print(f"决定系数: {r2:.2f}")
