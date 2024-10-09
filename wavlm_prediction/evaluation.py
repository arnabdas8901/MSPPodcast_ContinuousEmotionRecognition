import sys
import torch
import pandas as pd
sys.path.append("/home/adas/Projects/Wav_LM/")
from losses import CCCLoss
from model import TLTRModel
from data_loader import build_webdataloader

test_shad_urls = "/ds/audio/MSP_Podcast/wavlm_large_features/test-{000000..000061}.tar"
destination_path ="/netscratch/adas/experments/EmoRecog/wavlm_evaluation.csv"
model_mode = "tl_tr_512_1_8"
pretrained_model = "/netscratch/adas/experments/EmoRecog/WavLM_TLTR_noDownsample/epoch_00016.pth"
model = TLTRModel(mode=model_mode)
weights = torch.load(pretrained_model, map_location='cpu')['model']
model.load_state_dict(weights.state_dict())
model.eval()


val_dataloader = build_webdataloader(test_shad_urls,
                                      batch_size=1,
                                      validation=True,
                                      num_workers=8,
                                      device='cpu',
                                      return_key=True
                                      )

file_names = []
val_gts  = []
arou_gts = []
val_preds = []
arou_preds = []
for eval_steps_per_epoch, batch in enumerate(val_dataloader):
    x_feat, y_org, y_val, y_arou, y_dom, key_list = batch
    preds = model.forward(x_feat.float())
    val_pred = preds[:, 0]
    arou_pred = preds[:,1]
    val_gts.append(y_val.item())
    arou_gts.append(y_arou.item())
    val_preds.append(val_pred.detach().item())
    arou_preds.append(arou_pred.detach().item())
    key = key_list[0].split("/")[-1]
    print(key)
    file_names.append(key)

df = pd.DataFrame({'filename': file_names,
                   'valence GT': val_gts,
                   'arousal GT': arou_gts,
                   'valence pred': val_preds,
                   'arousal pred': arou_preds})
df.to_csv(destination_path, index=False)

ccc_loss = CCCLoss()
print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_preds).squeeze(), torch.tensor(val_gts).squeeze()))
print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_preds).squeeze(), torch.tensor(arou_gts).squeeze()))
print("End")