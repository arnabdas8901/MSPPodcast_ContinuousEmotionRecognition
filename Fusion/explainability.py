import os
import math
import sys
import copy
import torch
import warnings
import textgrid
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.functional as f
sys.path.append("/home/adas/Projects/Wav_LM/")
sys.path.append("/home/adas/Projects/Seamless/")
warnings.filterwarnings("ignore")
from model import  TLTRModel, FusionTLTRModel
from extract_features import _get_whisper_model, extract_save_whisper_features
from forced_alignment import *
from WavLM import WavLM, WavLMConfig
os.system('export TEMP=/netscratch/adas/Temp/')
from wavlm_prediction.extract_feature import extract_feature
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model
from extract_audio_features import extract_w2vbert_feature
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter



def filter_audio(audio_path, centre_f, range_f, save_original=False):
    wav, sr = torchaudio.load(audio_path)
    resampled_wav = f.resample(wav, sr, 16000, rolloff=0.99)
    filtered_wav = copy.deepcopy(resampled_wav)

    if not  save_original:
        for i in range(3):
            filtered_wav = f.bandpass_biquad(filtered_wav, 16000, centre_f, centre_f/range_f)
    torchaudio.save("/netscratch/adas/msp_filtered.wav", filtered_wav, 16000, format='wav')


def supress_wave(audio_path, xmin, x_max):
    xmin = math.floor(16000 * xmin)
    x_max = math.ceil(16000 * x_max)
    wav, sr = torchaudio.load(audio_path)
    resampled_wav = f.resample(wav, sr, 16000, rolloff=0.99)
    filtered_wav = copy.deepcopy(resampled_wav)
    filtered_wav[:, xmin:x_max] = 0
    torchaudio.save("/netscratch/adas/msp_filtered.wav", filtered_wav, 16000, format='wav')

def predict(whisper_model, whisper_TLTR_model):
    feat = extract_save_whisper_features(whisper_model, "/netscratch/adas/msp_filtered.wav", save=False)
    feat = feat.detach()[:,:30,:]

    prediction = whisper_TLTR_model.forward(feat.unsqueeze(0)).detach()
    return prediction.squeeze().numpy()

def predict_fusion(whisper_model, wavlm_model, w2v_model, fusion_model, wavlm_normalize):
    whisper_feat = extract_save_whisper_features(whisper_model, "/netscratch/adas/msp_filtered.wav", save=False)
    whisper_feat = whisper_feat.detach()[:, :30, :]

    wavlm_feat = extract_feature(wavlm_model, "/netscratch/adas/msp_filtered.wav", wavlm_normalize)
    wavlm_feat = wavlm_feat.detach()[:,:29,:]

    w2v_feat = extract_w2vbert_feature(w2v_model, "/netscratch/adas/msp_filtered.wav", audio_decoder, fbank_converter, collater)
    w2v_feat = w2v_feat.detach()[:,:30, :]


    prediction = fusion_model.forward(whisper_feat.unsqueeze(0), wavlm_feat.unsqueeze(0), w2v_feat.unsqueeze(0)).detach()
    return prediction.squeeze().numpy()

if __name__ == '__main__':
    freq_centres = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 100, 200, 300, 400, 500, 800, 1000]
    band_widths = [100, 100, 100, 110, 110, 120, 140, 150, 160, 190, 210, 240, 280, 320, 380, 450, 550, 200, 400, 600, 800, 1000, 1600, 2000]
    audio_path = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_0866_0074_0002.wav"
    pretrained_model = "/netscratch/adas/experments/EmoRecog/Fusion_noDownsample_TLTR_large_head_NoMixupLoss/epoch_00023.pth"
    vals = []
    arous = []
    labels = []

    whisper_model = _get_whisper_model()

    whisper_TLTR_model = TLTRModel(mode="tl_tr_512_1_8", n_layer=24)
    whisper_TLTR_path = "/netscratch/adas/experments/EmoRecog/Whisper_TLTR_noDownsample/epoch_00008.pth"
    whisper_TLTR_model.load_state_dict(
        torch.load(whisper_TLTR_path, map_location='cpu')['model'].state_dict())
    whisper_TLTR_model.eval()

    checkpoint_path = "/netscratch/adas/PretrainedModels/WavLM/WavLM-Large.pt"
    checkpoint = torch.load(checkpoint_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    wavlm_model = WavLM(cfg)
    wavlm_model.load_state_dict(checkpoint['model'])

    wavlm_TLTR_model = TLTRModel(mode="tl_tr_512_1_8", n_layer=25)
    wavlm_TLTR_model.load_state_dict(
        torch.load("/netscratch/adas/experments/EmoRecog/WavLM_TLTR_noDownsample/epoch_00016.pth", map_location='cpu')['model'].state_dict())
    wavlm_TLTR_model.eval()

    w2v_model = load_conformer_shaw_model("conformer_shaw", device=torch.device("cpu"), dtype=torch.float32, )
    w2v_model.eval()
    w2v_TLTR_model = TLTRModel(mode="tl_tr_512_1_8", n_layer=24)
    w2v_TLTR_model.load_state_dict(
        torch.load("/netscratch/adas/experments/EmoRecog/Seamless_TLTR_noDownsample/epoch_00039.pth", map_location='cpu')['model'].state_dict())
    w2v_TLTR_model.eval()

    model = FusionTLTRModel(whisper_TLTR_model, wavlm_TLTR_model, w2v_TLTR_model)
    weights = torch.load(pretrained_model, map_location='cpu')['model']
    model.load_state_dict(weights.state_dict())
    model.eval()

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


    """vals.append(0.0)
    arous.append(1.0)
    labels.append("GT")

    filter_audio(audio_path, None, None, save_original=True)
    #preds = predict(whisper_model, whisper_TLTR_model)
    preds = predict_fusion(whisper_model, wavlm_model, w2v_model, model, wavlm_normalize=cfg.normalize)
    vals.append(round(preds[0],2))
    arous.append(round(preds[1], 2))
    labels.append("All")

    for centre_f, bw in zip(freq_centres, band_widths):
        filter_audio(audio_path, centre_f, bw)
        #preds = predict(whisper_model, whisper_TLTR_model)
        preds = predict_fusion(whisper_model, wavlm_model, w2v_model, model, wavlm_normalize=cfg.normalize)
        vals.append(round(preds[0],2))
        arous.append(round(preds[1],2))
        labels.append(str(int(centre_f- bw/2))+"-"+str(int(centre_f + bw/2)))

    pred_dict = {
        #'Valence': vals,
        'Arousal': arous,
    }

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in pred_dict.items():
        offset = width * multiplier
        #rects = ax.bar(x + offset, measurement, width, label=attribute)
        rects = ax.bar(x, measurement, width, label=attribute, color='orange')
        ax.bar_label(rects, padding=3, rotation= -90)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Arousal')
    #ax.set_title('Valence-Arousal by frequency bands')
    #ax.set_xticks(x + width, labels, rotation = -90)
    ax.set_xticks(x, labels, rotation=-90)
    ax.set_xlabel('Frequency bands')
    #ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(0, 1.2)

    plt.savefig('/netscratch/adas/val_arou_fusion_bandbased_new_1.2.png')"""


    # Word based interpretability
    #audio_path = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_0866_0074_0002/ds/audio/MSP_Podcast/Audios/"
    #textGrid_path = "/ds/audio/MSP_Podcast/ForceAligned/MSP-PODCAST_0866_0074_0002.TextGrid"
    audio_path = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_1821_0089.wav"
    textGrid_path = "/ds/audio/MSP_Podcast/ForceAligned/MSP-PODCAST_1821_0089.TextGrid"
    #audio_path = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_0700_0150_0001.wav"
    #textGrid_path = "/ds/audio/MSP_Podcast/ForceAligned/MSP-PODCAST_0700_0150_0001.TextGrid"
    #audio_path = "/ds/audio/MSP_Podcast/Audios/MSP-PODCAST_2343_1356.wav"
    #textGrid_path = "/ds/audio/MSP_Podcast/ForceAligned/MSP-PODCAST_2343_1356.TextGrid"
    vals = []
    arous = []
    labels = []

    #vals.append(0.0)
    #vals.append(1.0)
    vals.append(1.0)
    arous.append(0.83)
    labels.append("GT")

    filter_audio(audio_path, None, None, save_original=True)
    preds = predict_fusion(whisper_model, wavlm_model, w2v_model, model, wavlm_normalize=cfg.normalize)
    labels.append("Pred")
    vals.append(round(preds[0], 2))
    arous.append(round(preds[1], 2))

    tg = textgrid.TextGrid.fromFile(textGrid_path)
    interval_tiers = tg.tiers[0]
    intervals = [interv for interv in interval_tiers.intervals if len(interv.mark) > 0]
    total_intervals = len(intervals)
    for pos, interval in enumerate(intervals):
        print(interval.mark, interval.minTime, interval.maxTime)
    for pos, interval in enumerate(intervals):
        if interval.mark is not None and len(interval.mark) > 0:
            #if pos == total_intervals-3 :
            if pos == total_intervals:
                break
            #print(interval.mark,interval.minTime, interval.maxTime)
            """labels.append(interval.mark+ " " + interval_tiers.intervals[pos+1].mark +
                          " " + interval_tiers.intervals[pos+2].mark + " " + interval_tiers.intervals[pos+3].mark)"""
            """labels.append(interval.mark + " " + intervals[pos + 1].mark +
                          " " + intervals[pos + 2].mark + " " + intervals[pos + 3].mark)"""
            labels.append(interval.mark)
            print(labels)
            """supress_wave(audio_path, interval.minTime, interval.maxTime + interval_tiers.intervals[pos+1].maxTime
                         + interval_tiers.intervals[pos+2].maxTime + interval_tiers.intervals[pos+3].maxTime)"""
            #supress_wave(audio_path, interval.minTime, intervals[pos+3].maxTime)
            supress_wave(audio_path, interval.minTime, interval.maxTime)
            #preds = predict(whisper_model, whisper_TLTR_model)
            preds =  predict_fusion(whisper_model, wavlm_model, w2v_model, model, wavlm_normalize=cfg.normalize)
            vals.append(round(preds[0], 2))
            arous.append(round(preds[1], 2))

    pred_dict = {
        'Valence': vals,
        #'Arousal': arous,
    }

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in pred_dict.items():
        offset = width * multiplier
        #rects = ax.bar(x + offset, measurement, width, label=attribute)
        rects = ax.bar(x, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, rotation=-90)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Valence')
    #ax.set_title('Valence prediction by 1-Gram occlusion')
    #ax.set_xticks(x + width, labels, rotation=-90)
    ax.set_xticks(x, labels, rotation=-90)
    ax.set_xlabel('Suppressed 1-Grams')
    #ax.legend(loc='upper left', ncol=1)
    ax.set_ylim(0, 1.1)

    plt.savefig('/netscratch/adas/val_arou_fusion_quadgrambased_new_0.8_latest_1gram.png')

