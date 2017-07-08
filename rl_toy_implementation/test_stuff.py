import numpy as np
import pandas as pd


actual_list = pd.read_csv("results/actual_value_revisit.csv")["url"].tolist()
pred_list = pd.read_csv("results/linear_dqn_results/test_value_revisit.csv")["url"].tolist()

print(sorted(actual_list)[:10])
print(sorted(pred_list)[:10])

# print(len(set(actual_list).intersection(set(pred_list))))