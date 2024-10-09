import os
import sys
sys.path.append("/home/adas/Projects/Wav_LM/")
import torch
import pandas as pd
from losses import CCCLoss

slang_list = ['fucking']
csv_path = "/netscratch/adas/experments/EmoRecog/fusion_largeHead_evaluation.csv"
trans_path = "/ds/audio/MSP_Podcast/Transcripts/"
data = pd.read_csv(csv_path)

val_gt_slang= []
val_gt = []
arou_gt_slang = []
arou_gt = []

val_pred_slang = []
val_pred = []
arou_pred_slang = []
arou_pred = []

for idx, row in data.iterrows():
    file_id = row['filename']
    trans_path_full = os.path.join(trans_path, file_id+".txt")
    content = open(trans_path_full, 'r').read()
    has_slang = [True if sl in content else False for sl in slang_list]
    val = float(row['valence GT'])
    arou = float(row['arousal GT'])
    if sum(has_slang) and val > 0.5:
        print(content + '\n', val, float(row['valence pred']))
        val_gt_slang.append(val)
        val_pred_slang.append(float(row['valence pred']))
        arou_gt_slang.append(arou)
        arou_pred_slang.append(float(row['arousal pred']))
    else:
        val_gt.append(float(row['valence GT']))
        val_pred.append(float(row['valence pred']))

        arou_gt.append(arou)
        arou_pred.append(float(row['arousal pred']))

ccc_loss = CCCLoss()
print("Val CCC Slang", ccc_loss.return_ccc(torch.tensor(val_pred_slang).squeeze(), torch.tensor(val_gt_slang).squeeze()))
print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))

print("Arou CCC Slang", ccc_loss.return_ccc(torch.tensor(arou_pred_slang).squeeze(), torch.tensor(arou_gt_slang).squeeze()))
print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))


