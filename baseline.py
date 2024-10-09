import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
from munch import Munch
import torch.nn.functional as F
from transformers import AutoProcessor, HubertModel, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from losses import CCCLoss
from typing import List
from transformers import Wav2Vec2Config


class HubertLarge(nn.Module):
    def __init__(self, feature_len, dim_out):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_len, 1024),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(1204, dim_out),
        )

    def forward(self, x):
        ip_features = self.processor(x, return_tensors="pt").input_values
        hidden_states = self.model.forward(ip_features, output_hidden_states=True).hidden_states[11]
        hidden_states = hidden_states.mean(dim=1)
        return self.regression_head(hidden_states)

    def train(self, mode: bool = True):
        super().train()
        self.model.feature_extractor.eval()
        self.model.feature_projection.eva()


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        #x = nn.ReLU()(x)

        return x


class RegressionHeadWithGender(RegressionHead):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size+2, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

class RegressionHeadWithGenderSpk(RegressionHead):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size+2+192, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

    def train(self, mode: bool = True):
        super().train()
        self.wav2vec2.freeze_feature_encoder()
        self.wav2vec2.feature_extractor.training = False
        self.classifier.train()


class EmotionModelWithGender(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHeadWithGender(config)
        self.sap = nn.Sequential(nn.Conv1d(1024 ,256, 7, 1,padding="same", bias=False),
                                 nn.SELU(),
                                 nn.InstanceNorm1d(256),
                                 nn.Conv1d(256 ,1, 7, 1,padding="same", bias=False),
                                 nn.Softmax(dim=-1))
        self.init_weights()

    def forward(
            self,
            input_values,
            gender
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        attn_weights = self.sap(torch.transpose(hidden_states, -1,-2))
        #hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = torch.sum(hidden_states * torch.transpose(attn_weights, -1, -2), dim=1)
        hidden_states = torch.cat([hidden_states, F.one_hot(gender, num_classes=2).float()], dim=-1)
        logits = self.classifier(hidden_states)

        return logits

    def train(self, mode: bool = True):
        super().train()
        self.wav2vec2.freeze_feature_encoder()
        self.wav2vec2.feature_extractor.training = False
        self.classifier.train()

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        del load


class EmotionModelWithGenderSpk(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHeadWithGenderSpk(config)
        self.sap = nn.Sequential(nn.Conv1d(1024 ,256, 7, 1,padding="same", bias=False),
                                 nn.SELU(),
                                 nn.InstanceNorm1d(256),
                                 nn.Conv1d(256 ,1, 7, 1,padding="same", bias=False),
                                 nn.Softmax(dim=-1))
        self.init_weights()

    def forward(
            self,
            input_values,
            gender,
            spk_embed
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        attn_weights = self.sap(torch.transpose(hidden_states, -1,-2))
        #hidden_states = torch.mean(hidden_states, dim=1)
        hidden_states = torch.sum(hidden_states * torch.transpose(attn_weights, -1, -2), dim=1)
        hidden_states = torch.cat([hidden_states, F.one_hot(gender, num_classes=2).float(), spk_embed], dim=-1)
        logits = self.classifier(hidden_states)

        return logits

    def train(self, mode: bool = True):
        super().train()
        self.wav2vec2.freeze_feature_encoder()
        self.wav2vec2.feature_extractor.training = False
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier.train()
        self.sap.train()

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        del load


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
    gender=None
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).unsqueeze(0).to(device)

    # run through model
    with torch.no_grad():
        if gender is None:
            y = model(y)[0 if embeddings else 1]
        else:
            y = model(y, gender)

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

if __name__ == '__main__':
    # load model from hub
    device = 'cpu'
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    #processor = Wav2Vec2Processor.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained("/home/adas/.cache/huggingface/hub/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/snapshots/58f18fe614ea6a3a87f0aee59d49bcac0e885dea/")
    """config_w2v2 = Wav2Vec2Config.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                                                 _from_auto=False,
                                                 _from_pipeline=None)
    model = EmotionModel(config_w2v2)"""
    """config_w2v2 = Wav2Vec2Config.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                                                 _from_auto=False,
                                                 _from_pipeline=None)
    config_w2v2._name_or_path = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    model = EmotionModelWithGender(config_w2v2)
    model.load_state_dict(torch.load("/netscratch/adas/experments/EmoRecog/W2V2_WithGender/epoch_00005.pth", map_location="cpu"),
                          strict=False)
    #os.makedirs("/netscratch/adas/experments/EmoRecog/Baseline/", exist_ok=True)"""
    #torch.save(model.state_dict(), "/netscratch/adas/experments/EmoRecog/Baseline/w2v2_modelWeights,pt")"""
    model = EmotionModel.from_pretrained("/home/adas/.cache/huggingface/hub/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/snapshots/58f18fe614ea6a3a87f0aee59d49bcac0e885dea/", use_safetensors=False)
    #model.load_state_dict(torch.load("/home/adas/.cache/huggingface/hub/models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/blobs/176d9d1ce29a8bddbab44068b9c1c194c51624c7f1812905e01355da58b18816"))

    ccc_loss = CCCLoss()
    sampling_rate = 16000

    val_gt = []
    arou_gt = []
    dom_gt = []
    val_pred = []
    arou_pred = []
    dom_pred = []
    file_names = []

    #Gender fetch
    label_file = "/ds/audio/MSP_Podcast/Labels/labels_consensus.json"
    file = open(label_file, "r")
    labels = json.loads(file.read())
    file.close()
    label_summary = Munch(labels)

    test_file = open("/home/adas/Projects/StarGAN_v2/Valenzia/Data/MSP_Podcast/test_list.txt", 'r')
    lines = test_file.readlines()

    destination_path = "/netscratch/adas/experments/EmoRecog/baseline_evaluate_usingSavedWeights_3.csv"

    for idx, line in enumerate(lines):
        test_file = line.split("|")[0]
        file_name = test_file.split("/")[-1]
        file_names.append(file_name.split(".")[0])
        val = float(line.split("|")[2])
        arou = float(line.split("|")[3])
        dom = float(line.split("|")[4])
        val_gt.append((val-1)/6 )
        arou_gt.append((arou - 1) / 6)
        dom_gt.append((dom- 1) / 6)
        signal, _ = librosa.load(test_file, sr=sampling_rate)
        signal = np.expand_dims(signal, axis=0)
        gender = Munch(label_summary[file_name]).Gender
        gender = 0 if gender == "Male" else 1
        #out = process_func(signal, sampling_rate, gender=torch.tensor(gender).unsqueeze(0))
        out = process_func(signal, sampling_rate)
        print(idx+1, file_name, "- Arou, dom , val:", out)
        val_pred.append(out[0][-1])
        arou_pred.append(out[0][0])
        dom_pred.append(out[0][1])
        # print(process_func(signal, sampling_rate, embeddings=True))

    df = pd.DataFrame({'filename': file_names,
                       'valence GT': val_gt,
                       'arousal GT': arou_gt,
                       'valene pred': val_pred,
                       'arousal pred': arou_pred})
    df.to_csv(destination_path, index=False)

    print("Val CCC", ccc_loss.return_ccc(torch.tensor(val_pred).squeeze(), torch.tensor(val_gt).squeeze()))
    print("Arou CCC", ccc_loss.return_ccc(torch.tensor(arou_pred).squeeze(), torch.tensor(arou_gt).squeeze()))
    print("Dom CCC", ccc_loss.return_ccc(torch.tensor(dom_pred).squeeze(), torch.tensor(dom_gt).squeeze()))




