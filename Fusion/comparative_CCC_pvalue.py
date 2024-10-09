import torch
import numpy as np
import pandas as pd
from scipy import stats
from losses import CCCLoss

csv_path = "/netscratch/adas/experments/EmoRecog/fusion_baseline_largeHead_evaluation.csv"
result = pd.read_csv(csv_path)
result = result.dropna()
result.sort_values(['filename'], inplace=True)

n_iterations=1000
val_stats = []
val_stats_baseline = []
arou_stats = []
arou_stats_baseline = []
ccc_loss = CCCLoss()
for i in range(n_iterations):
    result_sampled = result.sample(frac=0.5)
    val_gt = result_sampled['valence GT'].values.tolist()
    arou_gt = result_sampled['arousal GT'].values.tolist()
    val_pred = result_sampled['valence pred'].values.tolist()
    arou_pred = result_sampled['arousal pred'].values.tolist()
    val_pred_BL = result_sampled['base val'].values.tolist()
    arou_pred_BL = result_sampled['base arou'].values.tolist()
    val_stats.append(ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()).item())
    arou_stats.append(ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()).item())

    val_stats_baseline.append(ccc_loss.return_ccc(torch.tensor(val_pred_BL).squeeze(), torch.tensor(val_gt).squeeze()).item())
    arou_stats_baseline.append(ccc_loss.return_ccc(torch.tensor(arou_pred_BL).squeeze(), torch.tensor(arou_gt).squeeze()).item())

print(val_stats)
print(val_stats_baseline)

print(arou_stats)
print(arou_stats_baseline)

res = stats.ttest_rel(val_stats, val_stats_baseline, alternative="greater")
print("val", res)

res = stats.ttest_rel(arou_stats, arou_stats_baseline, alternative="greater")
print("arou", res)

import statistics
diff = [a-b for a, b in zip(arou_stats, arou_stats_baseline)]
print(statistics.mean(diff), statistics.stdev(diff), statistics.mean(diff)-statistics.stdev(diff))
