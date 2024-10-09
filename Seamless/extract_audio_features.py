import os
import librosa
os.environ['FAIRSEQ2_CACHE_DIR'] = "/netscratch/adas/Temp/"
import torch
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from pathlib import Path
import numpy as np
import soundfile as sf
from fairseq2.nn.transformer import EncoderLayerOutputHook
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model

class Hook(EncoderLayerOutputHook):
    def __init__(self):
        super().__init__()
        self.output = None
        self.handle = None
    def __call__(self, layer_idx,
        layer_output,
        layer_padding_mask,
        num_layers):
        self.output = layer_output
        self.handle.remove()
        return False


def extract_feature(files, dir_path):
    for line in files:
        file_name = line.split("|")[0]
        print(file_name)
        basename = os.path.basename(file_name).split('.')[0]
        if os.path.isfile(os.path.join(dir_path, basename + '.pt')):
            print("Feature Exists.")
            continue
        hooks = []
        for _, _ in enumerate(model.encoder.layers):
            hook = Hook()
            handle = model.encoder.register_layer_output_hook(hook)
            hook.handle = handle
            hooks.append(hook)

        #actual_wav = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_0001_0008.wav"
        actual_wav, sr = librosa.load(file_name, sr=16000)
        wav = np.zeros(16000*12)
        wav[:len(actual_wav)] = actual_wav
        sf.write("/netscratch/adas/blank.wav", wav, 16000, 'PCM_24')

        audio_wav_path, device, dtype = "/netscratch/adas/blank.wav", torch.device("cpu"), torch.float32
        #audio_wav_path, device, dtype = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_0001_0008.wav", torch.device("cpu"), torch.float32


        with Path(audio_wav_path).open("rb") as fb:
            block = MemoryBlock(fb.read())

        decoded_audio = audio_decoder(block)
        src = collater(fbank_converter(decoded_audio))["fbank"]
        seqs, padding_mask = get_seqs_and_padding_mask(src)

        with torch.inference_mode():
            seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
            seqs, padding_mask = model.encoder(seqs, padding_mask)

        seqs = [hk.output for hk in hooks]
        seqs = torch.stack(seqs, dim=0)
        seqs_padded = torch.zeros((24, 600, 1024), dtype=seqs.dtype)
        seqs_padded[:,:599, :] = seqs.squeeze()
        reps = torch.nn.functional.avg_pool2d(seqs_padded, kernel_size=(20, 1), stride=(20, 1))
        reps = reps.detach()
        reps.requires_gard = False
        torch.save(reps, os.path.join(dir_path, basename + '.pt'))

def extract_w2vbert_feature(model, line, audio_decoder, fbank_converter, collater):
    file_name = line.split("|")[0]
    #print(file_name)
    basename = os.path.basename(file_name).split('.')[0]
    dir_name = os.path.dirname(file_name).split("/")[-1]
    hooks = []
    for _, _ in enumerate(model.encoder.layers):
        hook = Hook()
        handle = model.encoder.register_layer_output_hook(hook)
        hook.handle = handle
        hooks.append(hook)

    # actual_wav = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_0001_0008.wav"
    actual_wav, sr = librosa.load(file_name, sr=16000)
    wav = np.zeros(16000 * 12)
    length = len(actual_wav)
    length = length if length <= 16000 * 12 else 16000 * 12
    wav[:length] = actual_wav[:length]
    sf.write("/home/adas/Projects/Seamless/temp_dir/"+ dir_name+ "_" +basename+"_fusion_large_blank.wav", wav, 16000, 'PCM_24')

    audio_wav_path, device, dtype = "/home/adas/Projects/Seamless/temp_dir/"+ dir_name+ "_" + basename+"_fusion_large_blank.wav", torch.device("cpu"), torch.float32
    # audio_wav_path, device, dtype = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_0001_0008.wav", torch.device("cpu"), torch.float32

    with Path(audio_wav_path).open("rb") as fb:
        block = MemoryBlock(fb.read())

    decoded_audio = audio_decoder(block)
    src = collater(fbank_converter(decoded_audio))["fbank"]
    seqs, padding_mask = get_seqs_and_padding_mask(src)

    with torch.inference_mode():
        seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
        seqs, padding_mask = model.encoder(seqs, padding_mask)

    seqs = [hk.output for hk in hooks]
    seqs = torch.stack(seqs, dim=0)
    #seqs_padded = torch.zeros((24, 600, 1024), dtype=seqs.dtype)
    #seqs_padded[:, :599, :] = seqs.squeeze()
    seqs_padded = torch.zeros((24, 600, 1024), dtype=seqs.dtype)
    seqs_padded[:, :599, :] = seqs.squeeze()
    reps = torch.nn.functional.avg_pool2d(seqs_padded, kernel_size=(20, 1), stride=(20, 1))
    reps = reps.detach()
    reps.requires_gard = False
    os.remove("/home/adas/Projects/Seamless/temp_dir/"+ dir_name+ "_"+ basename+"_fusion_large_blank.wav")
    return reps


if __name__ == '__main__':
    audio_decoder = AudioDecoder(dtype=torch.float32, device=torch.device("cpu"))
    fbank_converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2 ** 15,
        channel_last=True,
        standardize=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    collater = Collater(pad_value=1)

    model = load_conformer_shaw_model("conformer_shaw", device=torch.device("cpu"), dtype=torch.float32, )
    model.eval()

    train_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/train_list.txt"
    train_target = "/ds/audio/MSP_Podcast/w2vec_bert_features/train/"
    os.makedirs(train_target, exist_ok=True)
    lines = open(train_path, 'r').readlines()
    extract_feature(lines, train_target)
    print("End")