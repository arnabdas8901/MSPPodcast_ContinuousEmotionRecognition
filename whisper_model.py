import os
import torch
import librosa
import whisper
import torch.nn.functional as F
os.system('export TEMP=/netscratch/adas/Temp/')

def alternate_forward(encoder, x):
    x = x.unsqueeze(0)
    x = F.gelu(encoder.conv1(x))
    x = F.gelu(encoder.conv2(x))
    x = x.permute(0, 2, 1)

    assert x.shape[1:] == encoder.positional_embedding.shape, "incorrect audio shape"
    x = (x + encoder.positional_embedding).to(x.dtype)

    all_x = []
    for idx, block in enumerate(encoder.blocks):
        x = block(x)
        all_x.append(torch.nn.functional.avg_pool2d(x, kernel_size=(20, 1), stride=(20, 1))[0])
        print(idx)
    x = encoder.ln_post(x)
    all_x = torch.stack(all_x, dim=0)  # [num_layer, pooled_time, rep_dim], e.g., [32, 75, 1024]
    return x, all_x

whisper_model = whisper.load_model("medium.en", download_root='/netscratch/adas/Temp/')

audio, _ = librosa.load("/home/adas/Projects/StarGAN_v2/Data/p273/62.wav", sr=16000)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

op, op_feature = alternate_forward(whisper_model.encoder, mel)
print(op_feature.shape)
# decode the audio
#options = whisper.DecodingOptions(fp16=False, language="en")
#result = whisper.decode(whisper_model, mel, options)

# print the recognized text
#print(result.text)
print("End")