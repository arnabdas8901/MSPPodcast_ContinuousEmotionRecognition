import sys
import yaml
import torch
import pandas as pd
sys.path.append("/home/adas/Projects/Wav_LM/")
from model import FusionMultiTLTRModel
from data_loader import build_webdataloader


config = yaml.safe_load(open("/home/adas/Projects/Wav_LM/Fusion/config.yml"))
test_shad_urls = "/ds/audio/MSP_Podcast/whisper_wavlm_w2v/test-{000000..000239}.tar"
destination_path ="/netscratch/adas/experments/EmoRecog/fusion_multibranch_evaluation.csv"
model_mode = "tl_tr_512_1_8"
#pretrained_model = "/netscratch/adas/experments/EmoRecog/Fusion_noDownsample_TLTR_Trainable_NoMixupLoss/epoch_00010.pth"
pretrained_model = "/netscratch/adas/experments/EmoRecog/Fusion_noDownsample_TLTR_Muti_FromScratch_NoMixupLoss/epoch_00031.pth"


model = FusionMultiTLTRModel()
weights = torch.load(pretrained_model, map_location='cpu')['model']
model.load_state_dict(weights.state_dict())
model.eval()
print("Fusion models loaded")


val_dataloader = build_webdataloader(test_shad_urls,
                                      batch_size=1,
                                      validation=True,
                                      num_workers=12,
                                      device='cpu',
                                      return_key=True
                                      )

print("Data loader configured")
file_names = []
val_gts  = []
arou_gts = []
val_preds = []
arou_preds = []
print("Evaluation starts")
for eval_steps_per_epoch, batch in enumerate(val_dataloader):
    x_wsp_feat, x_wlm_feat, x_w2v_feat, y_org, y_val, y_arou, y_dom, key_list = batch
    preds = model.forward(x_wsp_feat.float(), x_wlm_feat.float(), x_w2v_feat.float())
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
print("End")