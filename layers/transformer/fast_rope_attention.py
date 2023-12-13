
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from .rotary_pos_encoding import RotaryPosEncoding
from .multi_head_attention import AttentionMask
import math


class FastRopeAttention(torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, input_size: Optional[int] = None,
                 projection_size: Optional[int] = None, output_size: Optional[int] = None,
                 rotate_fraction: float = 0.5, rope_base: float = 10000):

        super().__init__()

        self.input_size = input_size or state_size
        self.projection_size = projection_size or (state_size // n_heads)
        self.output_size = output_size or state_size
        self.n_rotate = int(rotate_fraction * self.projection_size)

        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)
        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x

        self.data_to_kv = torch.nn.Linear(self.input_size, 2 * n_heads * self.projection_size, bias=False)
        self.data_to_q = torch.nn.Linear(self.input_size, n_heads * self.projection_size, bias=False)
        self.out_proj = torch.nn.Linear(n_heads * self.projection_size, self.output_size, bias=False)
        self.reset_parameters()

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], -1, self.projection_size).transpose(-2, -3)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_rotate < self.projection_size:
            r_k = k[..., :self.n_rotate]
            nr_k = k[..., self.n_rotate:]
            r_q = q[..., :self.n_rotate]
            nr_q = q[..., self.n_rotate:]

            r_q, r_k = self.pe(r_q, r_k, offset)
            return torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            return self.pe(q, k, offset)

    def get_mask_tensor(self, src_len: int, mask: Optional[AttentionMask]) -> Optional[torch.Tensor]:
        if mask is None or (mask.position_mask is None and mask.src_length_mask is None):
            return None

        # mask.position_mask: [..., N_out, N_in]
        # mask.src_length_mask: [B, ...., N_in]
        # True where it has to be masked

        if mask.position_mask is not None:
            n_pad = src_len - mask.position_mask.shape[-1]
            if n_pad > 0:
                pm = F.pad(mask.position_mask, (n_pad, 0), 'constant', value=False)
            else:
                pm = mask.position_mask

        if mask.position_mask is None:
            m = mask.src_length_mask.unsqueeze(-2)
        elif mask.src_length_mask is None:
            m = pm
        else:
            m = mask.src_length_mask.unsqueeze(-2) | pm

        return ~m

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                pos_offset: Optional[int] = None, need_weights: bool = False):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]

        if pos_offset is None:
            assert curr_state.shape[1] == attend_to.shape[1], "If attend_to has different shape than curr_state, pos_offset should be provided"
            pos_offset = 0

        k, v = self.data_to_kv(attend_to).split(self.projection_size * self.n_heads, dim=-1)
        q = self.data_to_q(curr_state)
        q = self.dropout(q)

        k = self.project_to_torch_order(k)
        q = self.project_to_torch_order(q)
        v = self.project_to_torch_order(v)

        if self.n_rotate > 0:
            q, k = self.rotate(q, k, pos_offset or 0)

        # att = (q @ k.transpose(-2, -1) / math.sqrt(self.projection_size))
        # att.masked_fill_(~self.get_mask_tensor(attend_to.shape[-2], mask), float('-inf'))
        # att = F.softmax(att, dim=-1)
        # res = att @ v
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=self.get_mask_tensor(attend_to.shape[-2], mask))

        res = res.transpose(-2, -3).contiguous().view(*curr_state.shape[:-1], -1)
        return self.out_proj(res)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight)
