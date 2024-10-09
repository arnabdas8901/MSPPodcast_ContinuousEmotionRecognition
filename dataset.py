import random
import torch
import torchaudio
import json
from munch import Munch

import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader
#from encodec.utils import convert_audio
from transformers import Wav2Vec2Processor
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)


MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

# IEMOCAP
"""EMO_RANGE = {
    "val_max" : 5.5,
    "val_min" : 1.,
    "arou_max" : 5.,
    "arou_min" : 1.,
    "dom_max" : 5.,
    "dom_min" : 0.5
}"""

# MSP-Podcast
EMO_RANGE = {
    "val_max" : 7.,
    "val_min" : 1.,
    "arou_max" : 7.,
    "arou_min" : 1.,
    "dom_max" : 7.,
    "dom_min" : 1.
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=16000,
                 validation=False,
                 max_len = 12
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path, int(label), float(val), float(arou), float(dom)) for path, label, val, arou, dom in _data_list]
        self.data_list_per_class = {
            target: [(path, label , val, arou, dom) for path, label, val, arou, dom in self.data_list if label == target] \
            for target in list(set([label for _, label, _, _, _ in self.data_list]))}

        self.sr = sr
        self.validation = validation
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        wav, label, val, arou, dom, gender = self._load_data(data)
        return wav, label, val, arou, dom, gender

    def _load_data(self, path):
        wave_tensor, label, val, arou, dom, gender = self._load_tensor(path)
        wav = torch.tensor(wave_tensor)
        return wav, label, val, arou, dom, gender

    def _load_tensor(self, data):
        wave_path, label, val, arou, dom = data
        label = int(label)
        wave_name = wave_path.split("/")[-1]
        gender = Munch(label_summary[wave_name]).Gender
        gender = 0 if gender == "Male" else 1
        wave, sr = librosa.load(wave_path, sr=self.sr)
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)
        wave = preprocessor(wave, sampling_rate=self.sr)
        wave = wave['input_values'][0]
        if len(wave) > self.sr * self.max_len :
            random_start = np.random.randint(0, len(wave) - self.sr * self.max_len)
            wave = wave[:, random_start:random_start + self.sr * self.max_len]
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor, label, val, arou, dom, int(gender)


class Collater(object):
    def __init__(self, sr = 16000, max_len=12):
        self.sr = sr
        self.max_len = max_len

    def __call__(self, batch):
        batch_size = len(batch)
        wavs = torch.zeros((batch_size, self.sr*self.max_len)).float()
        labels = torch.zeros((batch_size)).long()
        genders = torch.zeros((batch_size)).long()
        vals = torch.zeros((batch_size)).float()
        arous = torch.zeros((batch_size)).float()
        doms = torch.zeros((batch_size)).float()

        for bid, (wav, label, val, arou, dom, gender) in enumerate(batch):
            seq_len = int(wav.size()[-1])
            wavs[bid][:seq_len] = wav
            labels[bid] = label
            genders[bid] = gender
            vals[bid] = (val - EMO_RANGE["val_min"]) / (EMO_RANGE['val_max'] - EMO_RANGE['val_min'])
            arous[bid] = (arou - EMO_RANGE["arou_min"]) / (EMO_RANGE['arou_max'] - EMO_RANGE['arou_min'])
            doms[bid] = (dom - EMO_RANGE["dom_min"]) / (EMO_RANGE['dom_max'] - EMO_RANGE['dom_min'])

        return wavs, labels, vals, arous, doms, genders

def build_dataloader(path_list,
                     validation=False,
                     batch_size=256,
                     num_workers=1,
                     device='cpu'
                     ):
    label_file = "/ds/audio/MSP_Podcast/Labels/labels_consensus.json"
    file = open(label_file, "r")
    labels = json.loads(file.read())
    file.close()
    global label_summary
    label_summary = Munch(labels)
    global preprocessor
    preprocessor = Wav2Vec2Processor.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')
    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater()
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
