import json
import torch
import random
import librosa
import logging
import numpy as np
import webdataset as wds
from munch import Munch
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from io import BytesIO

np.random.seed(1)
random.seed(1)

# MSP-Podcast
EMO_RANGE = {
    "val_max" : 7.,
    "val_min" : 1.,
    "arou_max" : 7.,
    "arou_min" : 1.,
    "dom_max" : 7.,
    "dom_min" : 1.
}


class WebDataCollater(object):
    def __init__(self, max_len=30, whisper_num_layers=24, wavlm_num_layers=25, w2v_num_layers=24, features = 1024, return_key=False):
        self.max_len = max_len
        self.whisper_num_layers = whisper_num_layers
        self.wavlm_num_layers = wavlm_num_layers
        self.w2v_num_layers = w2v_num_layers
        self.features = features
        self.return_key = return_key


    def __call__(self, batch):
        batch_size = len(batch)
        whisper_feats = torch.zeros((batch_size, self.whisper_num_layers, self.max_len, self.features)).float()
        wavlm_feats = torch.zeros((batch_size, self.wavlm_num_layers, self.max_len-1, self.features)).float()
        w2v_feats = torch.zeros((batch_size, self.w2v_num_layers, self.max_len, self.features)).float()
        labels = torch.zeros((batch_size)).long()
        vals = torch.zeros((batch_size)).float()
        arous = torch.zeros((batch_size)).float()
        doms = torch.zeros((batch_size)).float()
        if self.return_key:
            key_ist = []

        for bid, data_dict in enumerate(batch):
            whisper_feat = data_dict['whisper.pt']
            wavlm_feat = data_dict['wavlm.pt']
            w2v_feat = data_dict['w2v.pt']
            label = data_dict['cls']
            whisper_feats[bid] = whisper_feat[:,:30, :]
            wavlm_feats[bid] = wavlm_feat[:,:29, :]
            w2v_feats[bid] = w2v_feat[:,:30, :]
            labels[bid] = label[0]
            val = label[1]
            arou = label[2]
            dom = label[3]
            vals[bid] = (val - EMO_RANGE["val_min"]) / (EMO_RANGE['val_max'] - EMO_RANGE['val_min'])
            arous[bid] = (arou - EMO_RANGE["arou_min"]) / (EMO_RANGE['arou_max'] - EMO_RANGE['arou_min'])
            doms[bid] = (dom - EMO_RANGE["dom_min"]) / (EMO_RANGE['dom_max'] - EMO_RANGE['dom_min'])
            if self.return_key:
                key_ist.append(data_dict["__key__"])

        if self.return_key:
            return whisper_feats, wavlm_feats, w2v_feats, labels, vals, arous, doms, key_ist
        else:
            return whisper_feats, wavlm_feats, w2v_feats, labels, vals, arous, doms

def build_webdataloader(url,
                        validation=False,
                        batch_size=256,
                        num_workers=1,
                        device='cpu',
                        return_key = False
                        ):
    def process_data(x):
        x = torch.load(BytesIO(x))
        x.requires_grad = False
        x = x[:,:30,:]
        return x

    def process_labels(x):
        x = str(x, encoding='utf-8')
        x = x.split(",")
        x = [float(item) for item in x]
        return torch.tensor(x)

    dataset = wds.WebDataset(url, shardshuffle=(not validation)).shuffle(100)\
        .decode(wds.handle_extension("pt", lambda x: process_data(x)), wds.handle_extension("cls", lambda x: process_labels(x)))

    collate_fn = WebDataCollater(return_key=return_key)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader



if __name__ == '__main__':

    tar_url = "/ds/audio/MSP_Podcast/whisper_wavlm_w2v/train-000000.tar"
    dataloader = build_webdataloader(tar_url, num_workers=1, batch_size=2)
    for batch in dataloader:
        print("Loaded")

