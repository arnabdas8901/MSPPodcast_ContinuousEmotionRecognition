import torch
import torch.nn as nn
from WavLM import WavLM, WavLMConfig

class EmoWizard(nn.Module):
    def __init__(self, checkpoint_path, feature_len, dim_out):
        super().__init__()
        checkpoint = torch.load(checkpoint_path)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])

        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_len, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, dim_out),
        )

    def forward(self, x):
        if self.cfg.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
        rep = self.model.extract_features(x)[0]
        rep = rep.mean(dim = 1)

        return self.regression_head(rep)

    def train(self, mode: bool = True):
        super().train()
        self.model.feature_extractor.eval()

class ChannelAttentionBottleneck(nn.Module):
    def __init__(self, adaptive = False, channels = 96, gamma=2, beta = 1, sequence_length=128):
        super().__init__()
        if adaptive:
            self.kernel_size = (torch.ceil((channels/gamma) + (beta/gamma)) // 2 * 2 + 1).item()
        else:
            self.kernel_size = 11
        self.avg_pool = nn.AvgPool1d(sequence_length)
        self.conv = nn.Conv1d(1,1,self.kernel_size, padding="same", bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        pooled = self.avg_pool(x).transpose(-2, -1)
        conv_out = self.conv(pooled)
        conv_out = self.activation(conv_out)
        weights = nn.Softmax(dim= -1)(conv_out)

        return torch.matmul(weights, x).squeeze(1)
