import librosa
import torch
from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load('/netscratch/adas/PretrainedModels/WavLM/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

wav_path = "/netscratch/adas/00348.wav"
wave, _ = librosa.load(wav_path, sr=16000)
wav_input_16khz = torch.tensor(wave).unsqueeze(0)
if cfg.normalize:
    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
#rep = model.extract_features(wav_input_16khz)[0]
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]


print("End")