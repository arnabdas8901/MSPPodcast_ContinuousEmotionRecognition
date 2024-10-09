import torch
from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class FusionTLTRModel(nn.Module):
    def __init__(self, whisperTLTR, wavLMTLTR, w2vBERT_TLTR, label_dim=2, feature_len = 1024):
        super().__init__()
        self.whisper_tltr = whisperTLTR
        self.wavlm_tltr = wavLMTLTR
        self.w2vbert_tltr = w2vBERT_TLTR
        self.label_dim = label_dim
        self.rep_dim = feature_len
        #self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim * 3), nn.Linear(self.rep_dim * 3 , self.label_dim))
        self.mlp_layer = nn.Sequential(nn.Linear(self.rep_dim * 3, self.rep_dim * 3),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(self.rep_dim * 3, self.label_dim))

    def forward(self, audio_rep_whisper, audio_rep_wavlm, audio_rep_w2v):
        whisper_rep = self.whisper_tltr.get_feature(audio_rep_whisper)
        wavlm_rep = self.wavlm_tltr.get_feature(audio_rep_wavlm)
        w2v_rep = self.w2vbert_tltr.get_feature(audio_rep_w2v)

        audio_rep = torch.cat([whisper_rep, wavlm_rep, w2v_rep], dim=-1)
        pred = self.mlp_layer(audio_rep)

        return pred

class TLTRModel(nn.Module):
    def __init__(self, label_dim=2, n_layer=25, rep_dim=1024, mode='tl_down_tr_512_1_8'):
        super().__init__()
        self.mode = mode
        self.n_layer = n_layer
        self.rep_dim = rep_dim
        self.label_dim = label_dim

        # Time-and-Layer-Wise Transformer (TL-TR model)
        # tl_tr_1_8 = whisper original intermediate representation dim, 1-att-head time-transformer, 8-att-head layer transformer
        if 'tl_tr' in mode:
            self.num_tatt_head = int(mode.split('_')[-2])
            self.num_latt_head = int(mode.split('_')[-1])
            self.time_tr = ResidualAttentionBlock(self.rep_dim, self.num_tatt_head)
            self.layer_tr = ResidualAttentionBlock(self.rep_dim, self.num_latt_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.label_dim))

        # Time-and-Layer-Wise Transformer w/ Low-Dim Projection (TL-TR-512 model)
        # lw_down_tr_512_1_8 = 512-dim rep, 1-att-head time-transformer, 8-att-head layer transformer
        if 'tl_down_tr' in mode:
            self.inter_rep_dim = int(mode.split('_')[-3])
            self.num_tatt_head = int(mode.split('_')[-2])
            self.num_latt_head = int(mode.split('_')[-1])

            self.down_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.inter_rep_dim))
            self.time_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_tatt_head)
            self.layer_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_latt_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.inter_rep_dim), nn.Linear(self.inter_rep_dim, self.label_dim))

    def forward(self, audio_rep):
        # time resolution in seconds
        # input audio_rep in shape (#layer, #time steps, rep_dim), e.g., (25, 29, 1024) # for 12 seconds
        batch_size, num_layer, audio_len, rep_dim = audio_rep.shape[0], audio_rep.shape[1], audio_rep.shape[2] , audio_rep.shape[3]
        audio_rep = audio_rep.reshape([batch_size*num_layer, audio_len, rep_dim])

        # if lower intermediate representation dimension
        if 'tl_down_tr' in self.mode:
            audio_rep = self.down_layer(audio_rep.float())  # [25*b, 29, 512]
            rep_dim = self.inter_rep_dim
        audio_rep = self.time_tr(audio_rep)  # [25*b, 29, 512/1024]
        audio_rep = torch.mean(audio_rep, dim=1)  # [25*b, 512/1024]
        audio_rep = audio_rep.reshape([batch_size, num_layer, rep_dim]) #[b, 25, 512/1024]
        audio_rep = self.layer_tr(audio_rep)  # [b, 25, 512/1024]
        audio_rep = torch.mean(audio_rep, dim=1)  # [b, 512/1024]

        pred = self.mlp_layer(audio_rep.float())  # [b, 2]

        return pred

    def get_feature(self, audio_rep):
        batch_size, num_layer, audio_len, rep_dim = audio_rep.shape[0], audio_rep.shape[1], audio_rep.shape[2] , audio_rep.shape[3]
        audio_rep = audio_rep.reshape([batch_size*num_layer, audio_len, rep_dim])

        if 'tl_down_tr' in self.mode:
            audio_rep = self.down_layer(audio_rep.float())
            rep_dim = self.inter_rep_dim
        audio_rep = self.time_tr(audio_rep)
        audio_rep = torch.mean(audio_rep, dim=1)
        audio_rep = audio_rep.reshape([batch_size, num_layer, rep_dim])
        audio_rep = self.layer_tr(audio_rep)
        audio_rep = torch.mean(audio_rep, dim=1)

        return audio_rep

class FusionTLTRClassifierModel(nn.Module):
    def __init__(self, whisperTLTR, wavLMTLTR, w2vBERT_TLTR, label_dim=10, feature_len = 1024):
        super().__init__()
        self.whisper_tltr = whisperTLTR
        self.wavlm_tltr = wavLMTLTR
        self.w2vbert_tltr = w2vBERT_TLTR
        self.label_dim = label_dim
        self.rep_dim = feature_len
        self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim * 3), nn.Linear(self.rep_dim * 3 , self.label_dim))

    def forward(self, audio_rep_whisper, audio_rep_wavlm, audio_rep_w2v):
        whisper_rep = self.whisper_tltr.get_feature(audio_rep_whisper)
        wavlm_rep = self.wavlm_tltr.get_feature(audio_rep_wavlm)
        w2v_rep = self.w2vbert_tltr.get_feature(audio_rep_w2v)

        audio_rep = torch.cat([whisper_rep, wavlm_rep, w2v_rep], dim=-1)
        pred = self.mlp_layer(audio_rep)

        return pred

class FusionTLTRClassifierRegressorModel(nn.Module):
    def __init__(self, whisperTLTR, wavLMTLTR, w2vBERT_TLTR, label_dim=2, classifier_label_dim=10, feature_len = 1024):
        super().__init__()
        self.whisper_tltr = whisperTLTR
        self.wavlm_tltr = wavLMTLTR
        self.w2vbert_tltr = w2vBERT_TLTR
        self.label_dim = label_dim
        self.classifier_label_dim = classifier_label_dim
        self.rep_dim = feature_len
        self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim * 3), nn.Linear(self.rep_dim * 3 , self.label_dim))
        self.cls_mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim * 3), nn.Linear(self.rep_dim * 3, self.classifier_label_dim))

    def forward(self, audio_rep_whisper, audio_rep_wavlm, audio_rep_w2v):
        whisper_rep = self.whisper_tltr.get_feature(audio_rep_whisper)
        wavlm_rep = self.wavlm_tltr.get_feature(audio_rep_wavlm)
        w2v_rep = self.w2vbert_tltr.get_feature(audio_rep_w2v)

        audio_rep = torch.cat([whisper_rep, wavlm_rep, w2v_rep], dim=-1)
        pred = self.mlp_layer(audio_rep)
        cls_pred = self.cls_mlp_layer(audio_rep)

        return pred, cls_pred


class FusionMultiTLTRModel(nn.Module):
    def __init__(self, feature_len = 1024, mode="tl_tr_512_1_8"):
        super().__init__()
        self.whisper_tltr_val = TLTRModel(label_dim=1, n_layer=24, mode=mode)
        self.whisper_tltr_arou = TLTRModel(label_dim=1, n_layer=24, mode=mode)
        self.wavlm_tltr_val = TLTRModel(label_dim=1, n_layer=25, mode=mode)
        self.wavlm_tltr_arou = TLTRModel(label_dim=1, n_layer=25, mode=mode)
        self.w2vbert_tltr_val = TLTRModel(label_dim=1, n_layer=24, mode=mode)
        self.w2vbert_tltr_arou = TLTRModel(label_dim=1, n_layer=24, mode=mode)
        self.rep_dim = feature_len
        self.mlp_layer_val = nn.Sequential(nn.LayerNorm(self.rep_dim * 3), nn.Linear(self.rep_dim * 3 , 1))
        self.mlp_layer_arou = nn.Sequential(nn.LayerNorm(self.rep_dim * 3), nn.Linear(self.rep_dim * 3, 1))

    def forward(self, audio_rep_whisper, audio_rep_wavlm, audio_rep_w2v):
        whisper_rep_val = self.whisper_tltr_val.get_feature(audio_rep_whisper)
        wavlm_rep_val = self.wavlm_tltr_val.get_feature(audio_rep_wavlm)
        w2v_rep_val = self.w2vbert_tltr_val.get_feature(audio_rep_w2v)

        audio_rep_val = torch.cat([whisper_rep_val, wavlm_rep_val, w2v_rep_val], dim=-1)
        pred_val = self.mlp_layer_val(audio_rep_val)

        whisper_rep_arou = self.whisper_tltr_arou.get_feature(audio_rep_whisper)
        wavlm_rep_arou = self.wavlm_tltr_arou.get_feature(audio_rep_wavlm)
        w2v_rep_arou = self.w2vbert_tltr_arou.get_feature(audio_rep_w2v)

        audio_rep_arou = torch.cat([whisper_rep_arou, wavlm_rep_arou, w2v_rep_arou], dim=-1)
        pred_arou = self.mlp_layer_val(audio_rep_arou)

        return torch.cat([pred_val, pred_arou], dim=-1)