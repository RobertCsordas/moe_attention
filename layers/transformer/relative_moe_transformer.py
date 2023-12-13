
from typing import Optional, List, Union, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import ActivationFunction
from .multi_head_relative_pos_attention import FixedRelativeMultiheadAttention, AttentionMask
from .transformer_preln import reset_prenorm_params
from layers.moe_layer import MoE
import math
from framework import utils
from torch.profiler import profile, record_function, ProfilerActivity
from layers import LoggingLayer
from .full_moe_relative_attention import FullMoeRelativeAttention, FullMoeRopeAttention


class RelativeMoeTransformerEncoderLayer(LoggingLayer, torch.nn.Module):
    def __init__(self, d_model, nhead, n_experts: int, expert_size: int, n_layers: int,
                 dropout=0.1, activation: ActivationFunction = F.relu, attention_dropout=0,
                 test_pos_clamp: Optional[int] = None,
                 dropout_mode: str = "none", selection_mode: str = "add",
                 perplexity_reg: float = 0.0,
                 n_heads: int = 1, norm_keys: bool = False, perplexity_reg_mode: str="step",
                 n_random: int = 0, reg_type: str = "normal",
                 topk_mode: str = "full", head_projection_size: Optional[int] = None,
                 activation_after_topk: bool = False,
                 drop_parallel: bool = True,
                 normalize_expert_sel_init: bool = False, norm_key_init: bool = False, norm_value_init: bool = False,
                 identical_init: bool = False,
                 sel_norm: str = "none",
                 preln: bool = True, ln_affine: bool = True,
                 moe_dropout_factor: float = 1.0,
                 drop_expert: float = 0.0, sync_distributed: bool = True,
                 modulation_amplitude: float = 0.5, moe_init_scale: float = 1.0,
                 moe_att_n_experts: int = 4, moe_att_expert_dropout: Optional[float] = None,
                 moe_att_selection_mode: str = "sigmoid",
                 moe_att_k: Optional[int] = None, moe_att_ppl_reg: Optional[float] = None,
                 q_expert: bool = True, k_expert: bool = True, v_expert: bool = True,
                 o_expert: bool = True,
                 v_projection_size: Optional[int] = None,
                 qside_n_experts: Optional[int] = None,
                 moe_attention: bool = False, moe_att_variant: str = "full",
                 moe_att_shared_experts: bool = False,
                 moe_att_kq_n_experts: Optional[int] = None, moe_att_separate_kq_sel: bool = False,
                 moe_att_norm_init: bool = False, moe_att_same_sel: bool = False, moe_att_norm_retrieval: bool = False,
                 rotate_fraction: float = 0.5, rope_base: float = 10000):
        super().__init__()
        self.preln = preln
        self.i = 0

        if moe_attention:
            if moe_att_variant == "full":
                self.self_attn = FullMoeRelativeAttention(
                    d_model, nhead, dropout=attention_dropout,
                    projection_size=head_projection_size, init_std_scale=math.sqrt(2 / n_layers) if preln else 1.0,
                    n_experts=moe_att_n_experts,
                    perplexity_reg=perplexity_reg if moe_att_ppl_reg is None else moe_att_ppl_reg,
                    expert_dropout=drop_expert if moe_att_expert_dropout is None else moe_att_expert_dropout,
                    selection_mode=moe_att_selection_mode, q_expert=q_expert, k_expert=k_expert, v_expert=v_expert,
                    moe_k=n_heads if moe_att_k is None else moe_att_k, o_expert=o_expert, qside_n_experts=qside_n_experts,
                    v_projection_size=v_projection_size, shared_experts=moe_att_shared_experts,
                    kq_n_experts=moe_att_kq_n_experts, separate_kq_sel=moe_att_separate_kq_sel,
                    normalize_init=moe_att_norm_init,
                    same_sel=moe_att_same_sel, normalize_retrieval=moe_att_norm_retrieval,
                )
            elif moe_att_variant == "full_rope":
                self.self_attn = FullMoeRopeAttention(
                    d_model, nhead, dropout=attention_dropout,
                    projection_size=head_projection_size, init_std_scale=math.sqrt(2 / n_layers) if preln else 1.0,
                    n_experts=moe_att_n_experts,
                    perplexity_reg=perplexity_reg if moe_att_ppl_reg is None else moe_att_ppl_reg,
                    expert_dropout=drop_expert if moe_att_expert_dropout is None else moe_att_expert_dropout,
                    selection_mode=moe_att_selection_mode, q_expert=q_expert, k_expert=k_expert, v_expert=v_expert,
                    moe_k=n_heads if moe_att_k is None else moe_att_k, o_expert=o_expert, qside_n_experts=qside_n_experts,
                    v_projection_size=v_projection_size, shared_experts=moe_att_shared_experts,
                    kq_n_experts=moe_att_kq_n_experts, separate_kq_sel=moe_att_separate_kq_sel,
                    normalize_init=moe_att_norm_init, normalize_retrieval=moe_att_norm_retrieval,
                    rotate_fraction=rotate_fraction, rope_base=rope_base,
                )
            else:
                raise ValueError(f"Unknown attention variant {moe_att_variant}")
        else:
            self.self_attn = FixedRelativeMultiheadAttention(
                d_model, nhead, dropout=attention_dropout, test_pos_clamp=test_pos_clamp,
                projection_size=head_projection_size)

        std_scale = math.sqrt(2.0 / n_layers) if preln else 1.0
        std_scale *= math.sqrt(moe_init_scale)

        self.pkm = MoE(
            d_model, n_experts, expert_size, dropout=dropout * moe_dropout_factor, dropout_mode=dropout_mode,
            weight_scale=std_scale, selection_mode=selection_mode,
            perplexity_reg=perplexity_reg, n_heads=n_heads,
            norm_keys=norm_keys, perplexity_reg_mode=perplexity_reg_mode, n_random=n_random,
            reg_type=reg_type, topk_mode=topk_mode,
            activation_after_topk=activation_after_topk,
            activation=activation,
            normalize_expert_sel_init=normalize_expert_sel_init, norm_key_init=norm_key_init,
            norm_value_init=norm_value_init, identical_init=identical_init,
            sel_norm=sel_norm,
            expert_dropout=drop_expert,
            sync_distributed=sync_distributed,
            modulation_amplitude=modulation_amplitude)

        self.norm1 = torch.nn.LayerNorm(d_model, elementwise_affine=ln_affine)
        self.norm2 = torch.nn.LayerNorm(d_model, elementwise_affine=ln_affine)
        self.dropout = torch.nn.Dropout(dropout)

        self.activation = activation
        self.drop_parallel = drop_parallel

        if preln:
            reset_prenorm_params(self, n_layers)

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:

        src2 = self.norm1(src) if self.preln else src
        src2 = self.self_attn(src2, self.norm1(attend_to) if attend_to is not None else src2, mask,
                              pos_offset=pos_offset)
        src = src + self.dropout(src2)

        if self.preln:
            src2 = self.norm2(src)
        else:
            src = src2 = self.norm1(src)

        src3 = self.pkm(src2)

        src = src + self.dropout(src3)
        if not self.preln:
            src = self.norm2(src)
        return src
