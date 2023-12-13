import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention, AttentionMask
from typing import Optional, Callable, Dict, Type, Sequence, Union
from dataclasses import dataclass
# This file is based on PyTorch's internal implementation

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=attention_dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                      if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=attention_dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=attention_dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[AttentionMask] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                full_target: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:

        assert pos_offset == 0 or tgt_mask is None
        tgt2 = self.self_attn(tgt, tgt if full_target is None else full_target, mask=tgt_mask)
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


class TransformerDecoderBase(torch.nn.Module):
    layers: Union[Sequence[torch.nn.Module], torch.nn.ModuleList]

    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def create_state(self, batch_size: int, max_length: int, device: torch.device) -> State:
        return self.State(0, {i: torch.empty([batch_size, max_length, self.d_model], device=device)
                              for i in range(len(self.layers))})

    def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
        assert data.shape[1] == 1, f"For one-step forward should have one timesteps, but shape is {data.shape}"
        assert state.step < state.state[0].shape[1]

        for i, l in enumerate(self.layers):
            state.state[i][:, state.step:state.step + 1] = data
            data = l(data, *args, **kwargs, full_target=state.state[i][:, :state.step + 1],
                     pos_offset=state.step)

        state.step += 1
        return data


class TransformerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int, d_model, *args, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer(d_model, *args, **kwargs) for _ in range(n_layers)])
        pre_layernorm = hasattr(self.layers[-1], "is_preln") and self.layers[-1].is_preln
        self.output_map = torch.nn.LayerNorm(d_model) if pre_layernorm else lambda x: x

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for layer in self.layers:
            data = layer(data, *args, **kwargs)
        return self.output_map(data)


class TransformerDecoder(TransformerDecoderBase):
    def __init__(self, layer, n_layers: int, d_model: int, *args, **kwargs):
        super().__init__(d_model)
        self.layers = torch.nn.ModuleList([layer(d_model, *args, **kwargs) for _ in range(n_layers)])
        pre_layernorm = hasattr(self.layers[-1], "is_preln") and self.layers[-1].is_preln
        self.output_map = torch.nn.LayerNorm(d_model) if pre_layernorm else lambda x: x

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for layer in self.layers:
            data = layer(data, *args, **kwargs)
        return self.output_map(data)


def TransformerEncoderWithLayer(layer: Type[torch.nn.Module] = TransformerEncoderLayer):
    return lambda *args, **kwargs: TransformerEncoder(layer, *args, **kwargs)


def TransformerDecoderWithLayer(layer: Type[torch.nn.Module] = TransformerDecoderLayer):
    return lambda *args, **kwargs: TransformerDecoder(layer, *args, **kwargs)


class TransformerBase(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[AttentionMask] = None):

        memory = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, AttentionMask(None, tgt_mask), src_mask.src_length_mask if src_mask is not None else None)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


class Transformer(TransformerBase):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: ActivationFunction = F.relu,
                 encoder_layer=TransformerEncoderWithLayer(), decoder_layer=TransformerDecoderWithLayer(),
                 attention_dropout: float = 0):

        super().__init__(
            encoder_layer(num_encoder_layers, d_model, nhead, dim_feedforward, dropout, activation, attention_dropout),
            decoder_layer(num_decoder_layers, d_model, nhead, dim_feedforward, dropout, activation, attention_dropout))
