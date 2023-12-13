
from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_relative_pos_attention import AttentionMask
from .transformer import ActivationFunction
from .transformer_preln import reset_prenorm_params
from .full_moe_relative_attention import FullMoeRelativeAttention, FullMoeRopeAttention
from .moa import MoA
import math


class MoeAttentionRelativeTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, moe_att_n_experts, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0, drop_expand: bool = True,
                 head_projection_size: Optional[int] = None, preln: bool = False, n_layers: Optional[int] = None,
                 att_perplexity_reg: float = 0.0, expert_dropout: float = 0.0, att_selection_mode="sigmoid",
                 attention_variant="moa", q_expert: bool = True, k_expert: bool = True, v_expert: bool = True,
                 o_expert: bool = True, moe_k: int = 2,
                 norm_qk_score: bool = False, v_projection_size: Optional[int] = None, same_sel: bool = False,
                 qside_n_experts: Optional[int] = None, shared_experts: bool = False,
                 kq_n_experts: Optional[int] = None, separate_kq_sel: bool = False,
                 cvloss: float = 0.0, switchloss: float = 0.0, zloss: float = 0.0,
                 moa_mode: str = "my", rotate_fraction: float = 0.5, rope_base: float = 10000,
                 moeatt_norm_init: bool = False):
        super().__init__()
        self.is_preln = preln
        if attention_variant not in  {"full", "full_rope"} and (not q_expert):
            raise ValueError("q_expert can be disabled only when using qside attention")

        if attention_variant == "moa":
            self.self_attn = MoA(
                d_model, nhead, dropout=attention_dropout,
                projection_size=head_projection_size, init_std_scale=math.sqrt(2 / n_layers) if preln else 1.0,
                n_experts=moe_att_n_experts, perplexity_reg=att_perplexity_reg, expert_dropout=expert_dropout,
                selection_mode=att_selection_mode, mode=moa_mode, cvloss=cvloss, switchloss=switchloss, zloss=zloss
            )
        elif attention_variant == "full":
            self.self_attn = FullMoeRelativeAttention(
                d_model, nhead, dropout=attention_dropout,
                projection_size=head_projection_size, init_std_scale=math.sqrt(2 / n_layers) if preln else 1.0,
                n_experts=moe_att_n_experts, perplexity_reg=att_perplexity_reg, expert_dropout=expert_dropout,
                selection_mode=att_selection_mode, q_expert=q_expert, k_expert=k_expert, v_expert=v_expert,
                norm_qk_score=norm_qk_score, v_projection_size=v_projection_size, same_sel=same_sel,
                o_expert=o_expert, moe_k=moe_k, qside_n_experts=qside_n_experts,
                shared_experts=shared_experts, kq_n_experts=kq_n_experts, separate_kq_sel=separate_kq_sel,
                normalize_init=moeatt_norm_init
            )
        elif attention_variant == "full_rope":
            self.self_attn = FullMoeRopeAttention(
                d_model, nhead, dropout=attention_dropout,
                projection_size=head_projection_size, init_std_scale=math.sqrt(2 / n_layers) if preln else 1.0,
                n_experts=moe_att_n_experts, perplexity_reg=att_perplexity_reg, expert_dropout=expert_dropout,
                selection_mode=att_selection_mode, q_expert=q_expert, k_expert=k_expert, v_expert=v_expert,
                norm_qk_score=norm_qk_score, v_projection_size=v_projection_size, same_sel=same_sel,
                o_expert=o_expert, moe_k=moe_k, qside_n_experts=qside_n_experts,
                shared_experts=shared_experts, kq_n_experts=kq_n_experts, separate_kq_sel=separate_kq_sel,
                rotate_fraction=rotate_fraction, rope_base=rope_base,
                normalize_init=moeatt_norm_init
            )
        else:
            raise ValueError(f"Unknown attention variant: {attention_variant}")

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout) if drop_expand else lambda x: x
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation

        if preln:
            if n_layers is None:
                raise ValueError("n_layers must be specified when using preln")
            reset_prenorm_params(self, n_layers)
        else:
            self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:
        src2 = self.norm1(src) if self.is_preln else src
        src2 = self.self_attn(src2, self.norm1(attend_to) if attend_to is not None else src2, mask, pos_offset=pos_offset)
        src = src + self.dropout1(src2)

        if self.is_preln:
            src2 = self.norm2(src)
        else:
            src2 = src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if not self.is_preln:
            src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                     if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
