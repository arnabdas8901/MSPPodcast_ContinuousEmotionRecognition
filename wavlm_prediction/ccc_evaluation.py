import torch
import numpy as np
import pandas as pd
from losses import CCCLoss
import matplotlib.pyplot as plt

csv_path = "/netscratch/adas/experments/EmoRecog/whisper_evaluation_augmented.csv"
result = pd.read_csv(csv_path)
ccc_loss = CCCLoss()

val_gt = result['valence GT'].values.tolist()
arou_gt = result['arousal GT'].values.tolist()
val_pred = result['3_Avg_Val'].values.tolist()
arou_pred = result['3_Avg_Arou'].values.tolist()

def r(x,y):
    ''' Pearson Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rho = sxy / (np.std(x)*np.std(y))
    return rho

print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))
print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))

print("val rho", r(np.array(val_pred), np.array(val_gt)))
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
plt.show()