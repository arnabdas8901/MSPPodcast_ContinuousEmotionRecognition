import torch
import numpy as np
import pandas as pd
from losses import CCCLoss
import matplotlib.pyplot as plt
from statistics import mean, stdev

emotions_dict = {"N" : 0,
                 "A" : 1,
                 "S" : 2,
                 "U" : 3,
                 "C" : 4,
                 "H" : 5,
                 "F" : 6,
                 "D" : 7,
                 "O" : 8,
                 "X" : 9, }
emotions_dict = {int(y): x for x, y in emotions_dict.items()}

#csv_path = "/netscratch/adas/experments/EmoRecog/fusion_largeHead_evaluation.csv"
csv_path = "/netscratch/adas/experments/EmoRecog/fusion_large_withEmotion.csv"
result = pd.read_csv(csv_path)
ccc_loss = CCCLoss()

val_gt = result['valence GT'].values.tolist()
arou_gt = result['arousal GT'].values.tolist()
val_pred = result['valence pred'].values.tolist()
arou_pred = result['arousal pred'].values.tolist()
print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))
print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))

"""test_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/test_list.txt"
lines = open(test_path, 'r').readlines()


result["Emotion"] = ''
for idx, row in result.iterrows():
    filename = row['filename']
    print(filename)
    emo_clas = int([line for line in lines if filename in line][0][:-1].split('|')[1])
    print(result.loc[idx, 'filename'])
    result.loc[idx, 'Emotion'] = emotions_dict[emo_clas]"""

#result.to_csv("/netscratch/adas/experments/EmoRecog/fusion_large_withEmotion.csv", index=False)
#exit(0)

result = result.loc[result['Emotion'] == "X"]
val_gt = result['valence GT'].values.tolist()
#val_gt = [round(x, 2) for x in val_gt]
arou_gt = result['arousal GT'].values.tolist()
val_pred = result['valence pred'].values.tolist()
arou_pred = result['arousal pred'].values.tolist()
print(len(val_pred))
print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))
print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))

