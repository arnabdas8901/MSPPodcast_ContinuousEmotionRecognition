import torch
import pandas as pd
from losses import CCCLoss


csv_path = "/netscratch/adas/whisper_mosei.csv"
#csv_path = "/netscratch/adas/experments/EmoRecog/whisper_evaluation_augmented.csv"
result = pd.read_csv(csv_path)
ccc_loss = CCCLoss()

val_gt = result['valence GT'].values.tolist()
#arou_gt = result['arousal GT'].values.tolist()
val_pred = result['valence pred'].values.tolist()
#arou_pred = result['arousal pred'].values.tolist()



print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))
#print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))