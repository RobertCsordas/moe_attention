
from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_relative_pos_attention import AttentionMask
from .transformer import ActivationFunction, TransformerEncoder, TransformerDecoder, TransformerBase
import math
from .transformer import TransformerEncoderLayer, TransformerDecoderLayer


def reset_prenorm_params(m: torch.nn.Module, n_layers: int):
    for layer in m.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(math.sqrt(2 / (n_layers * layer.weight.shape[1])) / layer.weight.std())
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        elif isinstance(layer, torch.nn.LayerNorm):
            torch.nn.init.ones_(layer.weight)
            torch.nn.init.zeros_(layer.bias)


class PrelnTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, n_layers, dim_feedforward=2048, dropout=0.1,
                 activation: ActivationFunction = F.relu, attention_dropout=0):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, attention_dropout=attention_dropout)

        reset_prenorm_params(self, n_layers)

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, self.norm1(attend_to) if attend_to is not None else src2, mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class PrelnTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, n_layers, dim_feedforward=2048, dropout=0.1,
                 activation: ActivationFunction = F.relu, attention_dropout=0):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            attention_dropout=attention_dropout)

        reset_prenorm_params(self, n_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[AttentionMask] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                full_target: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
        assert pos_offset == 0 or tgt_mask is None
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2 if full_target is None else self.norm1(full_target), mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt2, memory, mask=AttentionMask(memory_key_padding_mask, None))
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class PrelnTransformerEncoder(TransformerEncoder):
    def __init__(self, layer, depth: int, d_model: int, *args, **kwargs):
        super().__init__(layer, depth, d_model, *args, **kwargs)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, *args, **kwargs):
        return self.norm(super().forward(*args, **kwargs))


class PrelnTransformerDecoder(TransformerDecoder):
    def __init__(self, layer, depth: int, d_model: int, *args, **kwargs):
        super().__init__(layer, depth, d_model, *args, **kwargs)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, *args, **kwargs):
        return self.norm(super().forward(*args, **kwargs))


class PreLnTransformer(TransformerBase):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: ActivationFunction = F.relu, attention_dropout: float = 0):

        super().__init__(
            PrelnTransformerEncoder(
                PrelnTransformerEncoderLayer, num_encoder_layers, d_model, nhead, num_encoder_layers,
                dim_feedforward, dropout, activation, attention_dropout),
            PrelnTransformerDecoder(
                PrelnTransformerDecoderLayer, num_decoder_layers, d_model, nhead, num_decoder_layers,
                dim_feedforward, dropout, activation, attention_dropout))
