import os
import sys
import math
import torch
import librosa
sys.path.append("/home/adas/Projects/Wav_LM/")
import pandas as pd
from losses import CCCLoss
from model import  TLTRModel
from extract_features import _get_whisper_model, extract_save_whisper_features


annotation_source_dirs = ["/ds/audio/IEMOCAP/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/Attribute/",
                          "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session2/dialog/EmoEvaluation/Attribute/",
                          "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session3/dialog/EmoEvaluation/Attribute/",
                          "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session4/dialog/EmoEvaluation/Attribute/",
                          "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session5/dialog/EmoEvaluation/Attribute/"]
wav_dirs =  ["/ds/audio/IEMOCAP/IEMOCAP_full_release/Session1/sentences/wav/",
             "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session2/sentences/wav/",
             "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session3/sentences/wav/",
             "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session4/sentences/wav/",
             "/ds/audio/IEMOCAP/IEMOCAP_full_release/Session5/sentences/wav/"]
destination_path = "/netscratch/adas/whisper_iemocap.csv"

whisper_model = _get_whisper_model()
whisper_TLTR_model = TLTRModel(mode="tl_tr_512_1_8", n_layer=24)
whisper_TLTR_path = "/netscratch/adas/experments/EmoRecog/Whisper_TLTR_noDownsample/epoch_00008.pth"
whisper_TLTR_model.load_state_dict(
    torch.load(whisper_TLTR_path, map_location='cpu')['model'].state_dict())
whisper_TLTR_model.eval()

total_count = 0
wav_paths = []
val_gt = []
arou_gt = []
val_pred = []
arou_pred = []
for annotation_source_dir, wav_dir in zip(annotation_source_dirs, wav_dirs):
    for files in os.listdir(annotation_source_dir):
        if files.endswith(".txt") and files.startswith("Ses"):
            complete_anot_file_path = os.path.join(annotation_source_dir, files)
            print(complete_anot_file_path)
            all_lines = open(complete_anot_file_path, "r").readlines()
            for line in all_lines:
                parts = line.split(":")
                identifier = parts[0].strip()
                wav_name = identifier+".wav"
                session_id = "_".join(identifier.split("_")[:-1])
                wav_path = os.path.join(wav_dir, session_id, wav_name)
                wav_paths.append(wav_path)
                print(wav_path)

                wav, _ = librosa.load(wav_path, sr= 16000)
                duration_sec = len(wav)/16000.
                frames = math.ceil(duration_sec * 2.5)

                act = float(parts[1].strip()[:-1].split(" ")[-1])
                arou_gt.append((act-1.)/5.)

                val = float(parts[2].strip()[:-1].split(" ")[-1])
                val_gt.append((val-1.)/5.)

                feat = extract_save_whisper_features(whisper_model, wav_path, save=False)
                feat = feat.detach()[:, :frames, :]
                prediction = whisper_TLTR_model.forward(feat.unsqueeze(0)).detach().squeeze()

                val_pd = round(prediction[0].item(), 2)
                arou_pd = round(prediction[1].item(), 2)
                #print(arou_pd, val_pd)

                arou_pred.append(arou_pd)
                val_pred.append(val_pd)

                total_count += 1


ccc_loss = CCCLoss()
print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))
print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))

print("Total wav processed", total_count)


df = pd.DataFrame({'filename': wav_paths,
                   'valence GT': val_gt,
                   'arousal GT': arou_gt,
                   'valence pred': val_pred,
                   'arousal pred': arou_pred})
df.to_csv(destination_path, index=False)
print("End")