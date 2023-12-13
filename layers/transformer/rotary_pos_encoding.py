import torch
import torch.jit
from typing import Tuple

# Based on: https://www.kaggle.com/code/aeryss/rotary-postional-encoding-rope-pytorch

# rotary pos emb helpers:
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


def apply_rot(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, seq_dim: int, offset: int) -> torch.Tensor:
    sin = sin.narrow(seq_dim, offset, x.shape[seq_dim])
    cos = cos.narrow(seq_dim, offset, x.shape[seq_dim])
    return (x * cos) + (rotate_half(x) * sin)

# [seq, batch, heads, hdim]
@torch.jit.script
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, seq_dim: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return apply_rot(q, sin, cos, seq_dim, offset), apply_rot(k, sin, cos, seq_dim, 0)


class RotaryPosEncoding(torch.nn.Module):
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = seq_dim

    def get(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[self.seq_dim]
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]

            self.cos_cached = emb.cos().view(*tgt_shape)
            self.sin_cached = emb.sin().view(*tgt_shape)

        return self.sin_cached, self.cos_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.get(k)
        return apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)
