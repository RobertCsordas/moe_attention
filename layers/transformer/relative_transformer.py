
from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import ActivationFunction
from .multi_head_relative_pos_attention import FixedRelativeMultiheadAttention, AttentionMask
from .multi_head_attention import MultiHeadAttention
from .transformer import Transformer, TransformerEncoderWithLayer, TransformerDecoderWithLayer


class RelativeTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0, test_pos_clamp: Optional[int] = None, drop_expand: bool = True,
                 head_projection_size: Optional[int] = None, ln_after_attention: bool = True):
        super().__init__()
        self.ln_after_attention = ln_after_attention
        self.self_attn = FixedRelativeMultiheadAttention(
            d_model, nhead, dropout=attention_dropout, test_pos_clamp=test_pos_clamp,
            projection_size=head_projection_size)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout) if drop_expand else lambda x: x
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        if ln_after_attention:
            self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:
        src2 = self.self_attn(src, attend_to if attend_to is not None else src, mask, pos_offset=pos_offset)
        src = src + self.dropout1(src2)
        src = self.norm1(src) if self.ln_after_attention else src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                     if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class RelativeTransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0, drop_expand: bool = True):
        super().__init__()

        self.self_attn = FixedRelativeMultiheadAttention(d_model, nhead, dropout=attention_dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=attention_dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout) if drop_expand else lambda x: x
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[AttentionMask] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                full_target: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
        assert pos_offset == 0 or tgt_mask is None
        tgt2 = self.self_attn(tgt, tgt if full_target is None else full_target, mask=tgt_mask,
                              pos_offset=pos_offset)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, mask=AttentionMask(memory_key_padding_mask, None))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                      if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class RelativeTransformer(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: ActivationFunction = F.relu, attention_dropout: float = 0):

        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation,
                         TransformerEncoderWithLayer(RelativeTransformerEncoderLayer),
                         TransformerDecoderWithLayer(RelativeTransformerDecoderLayer), attention_dropout)
