import os
os.system("export TEMP=/netscratch/adas/Temp/")
import librosa
from transformers import AutoProcessor, AutoFeatureExtractor, \
    HubertModel, Wav2Vec2Model, WavLMModel, AutoModel
model_list = ["facebook/wav2vec2-large-960h", "facebook/wav2vec2-large-960h-lv60", "facebook/wav2vec2-base-960h",
              "facebook/hubert-base-ls960", "microsoft/wavlm-base", "microsoft/wavlm-base-plus",
              "microsoft/wavlm-large"]

w2v2_large = AutoModel.from_pretrained(model_list[0],cache_dir="/netscratch/adas/Temp/")
wave_path = "/netscratch/adas/00348.wav"
wave, sr = librosa.load(wave_path, sr=16000)
if sr != 16000:
    wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
op = w2v2_large.feature_extractor(wave)
print("End")