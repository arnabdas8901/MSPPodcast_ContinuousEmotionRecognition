import torch
import numpy as np
import pandas as pd
from losses import CCCLoss
import matplotlib.pyplot as plt

csv_path = "/netscratch/adas/experments/EmoRecog/fusion_largeHead_evaluation.csv"
result = pd.read_csv(csv_path)
ccc_loss = CCCLoss()
result = result.dropna()
print(result.keys())
val_gt = result['valence GT'].values.tolist()
#val_gt = [round(x, 2) for x in val_gt]
arou_gt = result['arousal GT'].values.tolist()
val_pred = result['valence pred'].values.tolist()
arou_pred = result['arousal pred'].values.tolist()

def r(x,y):
    ''' Pearson Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rho = sxy / (np.std(x)*np.std(y))
    return rho

val_ccc = ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze())
arou_ccc = ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze())
print("Val CCC", val_ccc)
print("Arou CCC", arou_ccc)

n_iterations=1000
val_stats = []
arou_stats = []
for i in range(n_iterations):
    result_sampled = result.sample(frac=0.5)
    val_gt = result_sampled['valence GT'].values.tolist()
    arou_gt = result_sampled['arousal GT'].values.tolist()
    val_pred = result_sampled['valence pred'].values.tolist()
    arou_pred = result_sampled['arousal pred'].values.tolist()
    val_stats.append(ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))
    arou_stats.append(ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))


alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower_val = max(0.0, np.percentile(val_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper_val = min(1.0, np.percentile(val_stats, p))

print("Val", lower_val, upper_val)
print(val_ccc-lower_val, upper_val-val_ccc)

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower_arou = max(0.0, np.percentile(arou_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper_arou = min(1.0, np.percentile(arou_stats, p))
print("Arou", lower_arou, upper_arou)
print(arou_ccc-lower_arou, upper_arou-arou_ccc)


"""print("val rho", r(np.array(val_pred), np.array(val_gt)))
print("val rho", r(np.array(arou_pred), np.array(arou_gt)))

from scipy import stats
res = stats.kendalltau(np.array(val_pred), np.array(val_gt))
print("Ktau val", res.statistic)

res = stats.kendalltau(np.array(arou_pred), np.array(arou_gt))
print("Ktau arou", res.statistic)

plt.scatter(np.array(val_gt), np.array(val_pred))
plt.show()

plt.scatter(np.array(arou_gt), np.array(arou_pred))
plt.show()


plt.hist(np.array(val_gt), bins=10)
plt.show()
plt.hist(np.array(arou_gt), bins=10)
plt.show()"""