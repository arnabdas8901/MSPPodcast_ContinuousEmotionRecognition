import os
import sys
sys.path.append("/home/adas/Projects/Wav_LM/")
import torch
import librosa
import numpy as np
from WavLM import WavLM, WavLMConfig
os.system('export TEMP=/netscratch/adas/Temp/')





def save_feature(line, dir_path, save= True):
    file_name = line.split("|")[0]
    print(file_name)
    basename = os.path.basename(file_name).split('.')[0]
    #print(basename)
    wave = np.zeros(16000 * 12)
    source, _ = librosa.load(file_name, sr=16000)
    length = len(source)
    wave[:length] = source
    wave = torch.tensor(wave).float().unsqueeze(0)
    if cfg.normalize:
        wave = torch.nn.functional.layer_norm(wave, wave.shape)
    _, layer_results = model.extract_features(wave, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    layer_reps = [x.transpose(0, 1).detach() for x, _ in layer_results]
    reps = torch.stack(layer_reps, dim=0).squeeze()
    reps = torch.nn.functional.avg_pool2d(reps, kernel_size=(20, 1), stride=(20, 1))
    reps = reps.detach()
    reps.requires_gard = False
    if save:
        torch.save(reps, os.path.join(dir_path, basename+'.pt'))
        return None
    else:
        return reps


def extract_feature(model, line, normalize):
    file_name = line.split("|")[0]
    print(file_name)
    basename = os.path.basename(file_name).split('.')[0]
    #print(basename)
    wave = np.zeros(16000 * 12)
    source, _ = librosa.load(file_name, sr=16000)
    length = len(source)
    length = length if length <= 16000*12 else 16000*12
    wave[:length] = source[:length]
    wave = torch.tensor(wave).float().unsqueeze(0)
    if normalize:
        wave = torch.nn.functional.layer_norm(wave, wave.shape)
    _, layer_results = model.extract_features(wave, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    layer_reps = [x.transpose(0, 1).detach() for x, _ in layer_results]
    reps = torch.stack(layer_reps, dim=0).squeeze()
    reps = torch.nn.functional.avg_pool2d(reps, kernel_size=(20, 1), stride=(20, 1))
    reps = reps.detach()
    reps.requires_gard = False

    return reps


if __name__ == '__main__':
    checkpoint_path = "/netscratch/adas/PretrainedModels/WavLM/WavLM-Large.pt"
    checkpoint = torch.load(checkpoint_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])


    train_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/train_list.txt"
    train_target = "/ds/audio/MSP_Podcast/wavlm_large_features/train/"
    lines = open(train_path, 'r').readlines()
    for line in lines:
        save_feature(line, train_target)

    test_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/test_list.txt"
    test_target = "/ds/audio/MSP_Podcast/wavlm_large_features/test/"
    lines = open(test_path, 'r').readlines()
    for line in lines:
        save_feature(line, test_target)