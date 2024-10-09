import os
import sys
import copy
import warnings
import pandas as pd
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

def filter_audio(audio_path, centre_f, range_f, save_path):
    wav, sr = torchaudio.load(audio_path)
    resampled_wav = f.resample(wav, sr, 16000, rolloff=0.99)
    filtered_wav = copy.deepcopy(resampled_wav)

    for i in range(3):
        filtered_wav = f.bandpass_biquad(filtered_wav, 16000, centre_f, centre_f/range_f)
    torchaudio.save(save_path, filtered_wav, 16000, format='wav')


def predict_fusion(filepath, whisper_model, wavlm_model, w2v_model, fusion_model, wavlm_normalize):
    whisper_feat = extract_save_whisper_features(whisper_model, filepath, save=False)
    whisper_feat = whisper_feat.detach()[:, :30, :]

    wavlm_feat = extract_feature(wavlm_model, filepath, wavlm_normalize)

    w2v_feat = extract_w2vbert_feature(w2v_model, filepath, audio_decoder, fbank_converter, collater)


    prediction = fusion_model.forward(whisper_feat.unsqueeze(0), wavlm_feat.unsqueeze(0), w2v_feat.unsqueeze(0)).detach()
    return prediction.squeeze().numpy()

if __name__ == '__main__':
    pretrained_model = "/netscratch/adas/experments/EmoRecog/Fusion_noDownsample_TLTR_large_head_NoMixupLoss/epoch_00023.pth"
    test_path = "/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/test_list.txt"
    target_path = "/ds/audio/MSP_Podcast/pre_processed_test/"
    os.makedirs(target_path, exist_ok=True)
    destination_path = "/netscratch/adas/experments/EmoRecog/fusion_large_preprocess.csv"
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
        torch.load("/netscratch/adas/experments/EmoRecog/WavLM_TLTR_noDownsample/epoch_00016.pth", map_location='cpu')[
            'model'].state_dict())
    wavlm_TLTR_model.eval()

    w2v_model = load_conformer_shaw_model("conformer_shaw", device=torch.device("cpu"), dtype=torch.float32, )
    w2v_model.eval()
    w2v_TLTR_model = TLTRModel(mode="tl_tr_512_1_8", n_layer=24)
    w2v_TLTR_model.load_state_dict(
        torch.load("/netscratch/adas/experments/EmoRecog/Seamless_TLTR_noDownsample/epoch_00039.pth",
                   map_location='cpu')['model'].state_dict())
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

    lines = open(test_path, 'r').readlines()

    file_names = []
    val_preds = []
    arou_preds = []
    print("Evaluation starts")
    for line in lines:
        file_path = line.split("|")[0]
        file_idicator = os.path.basename(file_path)
        print(file_idicator)
        basename = os.path.basename(file_path).split(".")[0]
        target_file_path = os.path.join(target_path, file_idicator)
        filter_audio(file_path, 300, 600, target_file_path)
        prediction = predict_fusion(target_file_path, whisper_model, wavlm_model, w2v_model, model, cfg.normalize)
        val_pd = round(prediction[0].item(), 4)
        val_preds.append(val_pd)
        arou_pd = round(prediction[1].item(), 4)
        arou_preds.append(arou_pd)
        file_names.append(file_idicator)

    df = pd.DataFrame({'filename': file_names,
                       'valence pred': val_preds,
                       'arousal pred': arou_preds})
    df.to_csv(destination_path, index=False)

    print("End")


