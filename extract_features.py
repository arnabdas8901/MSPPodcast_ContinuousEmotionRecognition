import os
import torch
import librosa
import whisper
import functools
import torch.nn.functional as F
from torch.multiprocessing import Pool, get_context, set_start_method


def whisper_alternate_forward(encoder, x, avg_pool=True):
    x = x.unsqueeze(0)
    x = F.gelu(encoder.conv1(x))
    x = F.gelu(encoder.conv2(x))
    x = x.permute(0, 2, 1)

    assert x.shape[1:] == encoder.positional_embedding.shape, "incorrect audio shape"
    x = (x + encoder.positional_embedding).to(x.dtype)

    all_x = []
    for idx, block in enumerate(encoder.blocks):
        x = block(x)
        if avg_pool:
            all_x.append(torch.nn.functional.avg_pool2d(x, kernel_size=(20, 1), stride=(20, 1))[0])
        else:
            all_x.append(x[0])
    x = encoder.ln_post(x)
    all_x = torch.stack(all_x, dim=0)  # [num_layer, pooled_time, rep_dim], e.g., [24, 75, 1024]
    return x, all_x

def _get_whisper_model(model_type = "medium.en", download_root = '/netscratch/adas/Temp/'):
    ws_model = whisper.load_model(model_type, download_root=download_root)
    return ws_model

def extract_save_whisper_features(whisper_model, audio_path, sample_rate = 16000, save=True, avg_pool= True):
    filename = audio_path.split("/")[-1].split(".")[0]
    print(filename)
    if os.path.isfile("/ds-slt/audio/MSP_Podcast/whisper_medium_features/"+filename+".pt") :
        print("File present")
        return
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    _, op_feature = whisper_alternate_forward(whisper_model.encoder, mel, avg_pool)
    if save:
        torch.save(op_feature.detach().cpu(), "/ds-slt/audio/MSP_Podcast/whisper_medium_features/"+filename+".pt")
        return None
    else:
        return op_feature


if __name__ == '__main__':
    os.system('export TEMP=/netscratch/adas/Temp/')
    audio_dir = "/ds/audio/MSP_Podcast/Audios/"
    os.makedirs("/ds-slt/audio/MSP_Podcast/whisper_medium_features/", exist_ok=True)
    #set_start_method("spawn")
    whisper_model = _get_whisper_model()
    whisper_model = whisper_model.eval()
    #whisper_model = whisper_model.cuda()
    print("Model Loaded")
    """for files in os.listdir(audio_dir):
        complete_path = os.path.join(audio_dir, files)
        _ = extract_save_whisper_features(whisper_model, complete_path, save=True, avg_pool=False)"""
    complete_paths = [os.path.join(audio_dir, files) for files in os.listdir(audio_dir)]
    #pool = get_context("spawn").Pool(10)
    _ = [extract_save_whisper_features(whisper_model, f, save=True, avg_pool=False) for f in complete_paths ]
    #op = pool.map(functools.partial(extract_save_whisper_features, whisper_model= whisper_model, save=True, avg_pool=False), complete_paths)