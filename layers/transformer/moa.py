
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn.functional as F
from .multi_head_attention import AttentionMask
import framework
import math
from ..cvmm import cvmm, cvmm_prepare_sel, CVMMSel
from layers.regularized_layer import RegularizedLayer
from layers.once_per_iter_layer import OncePerIterLayer
from layers.logging_layer import LoggingLayer
import framework
from framework import utils


# Reimplementation of Mixture of Attention Heads: Selecting Attention Heads Per Token, with my (optionally) MoE
# Standard regularizations taken form the original code:
# https://github.com/yikangshen/MoA/blob/master/moa_layer/parallel_linear/moe.py


class MoA(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0, input_size: Optional[int] = None,
                 projection_size: Optional[int] = None, output_size: Optional[int] = None, init_std_scale: float = 1.0,
                 perplexity_reg: float = 0, share_pk: bool = True, expert_dropout: float = 0.0,
                 selection_mode: str = "sigmoid", mode: str = "my",
                 cvloss: float = 0.0, switchloss: float = 0.0, zloss: float = 0.0):

        super().__init__()

        self.input_size = input_size or state_size
        self.output_size = output_size or state_size
        self.pe_size = self.input_size
        self.n_experts = n_experts
        self.perplexity_reg = perplexity_reg
        self.sel_hist_dst = []
        self.share_pk = share_pk
        self.expert_dropout = expert_dropout
        self.selection_mode = selection_mode
        self.iter = 0
        self.sel_counts_dst_100 = 0
        self.mode = mode

        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss

        if self.mode not in {"my", "moa"}:
            raise ValueError("Unknown mode: " + self.mode)

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = projection_size or (state_size // n_heads)


        std_in = init_std_scale * math.sqrt(1 / self.input_size)
        std_out = init_std_scale * math.sqrt(1 / (n_heads * self.projection_size))
        std_pos = init_std_scale * math.sqrt(1 / self.pe_size)
        self.data_to_q = torch.nn.Parameter(torch.randn(n_experts, self.input_size, self.projection_size) * std_in)
        self.data_to_kv = torch.nn.Parameter(torch.randn(self.input_size, self.projection_size*2) * std_in)
        self.out_proj = torch.nn.Parameter(torch.randn(n_experts, self.projection_size, self.output_size) * std_out)
        self.pos_to_pk = torch.nn.Parameter(torch.randn(self.projection_size, self.pe_size) * std_pos)

        self.sel_dst = torch.nn.Parameter(torch.randn(n_experts, self.input_size) * std_in)

        self.renorm_rows(self.sel_dst)

        self.scale = torch.nn.Parameter(torch.full([1], 1.0 / math.sqrt(self.projection_size)))

        self.register_buffer("pos_encoding", self.create_pos_buffer(1000), persistent=False)

    def cv_squared(self, x):
        eps = 1e-10

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
            F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.n_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def create_pos_buffer(self, max_len: int):
        res = framework.layers.sinusoidal_pos_embedding(self.pe_size, 2 * max_len - 1, -max_len + 1,
                                                         device=self.pos_to_pk.device)

        assert res.shape[0] == 2 * max_len - 1
        return res

    def get_pos_subset(self, length: int, offset: int) -> torch.Tensor:
        total_len = length + offset
        if (2 * total_len - 1) > self.pos_encoding.shape[0]:
            self.pos_encoding = self.create_pos_buffer(total_len).to(self.pos_encoding.device).type_as(self.pos_encoding)

        return self.pos_encoding.narrow(0, self.pos_encoding.shape[0] // 2 - length + 1 - offset, 2 * length - 1)

    def project_to_torch_order(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        return x.view(*x.shape[:-1], -1, self.projection_size).transpose(-2, -3)

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

        return m

    def shift(self, posmat: torch.Tensor) -> torch.Tensor:
        # shape: [..., n_out, n_in * 2 - 1]
        # return: [..., n_out, n_in]

        n_in = (posmat.shape[-1] + 1) // 2
        n_neg = n_in - 1
        n_out = posmat.shape[-2]

        assert posmat.shape[-1] == n_in+n_neg

        # example:
        #p0[-3], p0[-2], p0[-1], | p0[0], p0[1], p0[2], p0[3] |
        #p1[-3], p1[-2], | p1[-1], p1[0], p1[1], p1[2],| p1[3]
        #p2[-3], |p2[-2], p2[-1], p2[0], p2[1],| p2[2], p2[3]
        #|p3[-3], p3[-2], p3[-1], p3[0],| p3[1], p3[2], p3[3]

        posmat = posmat.flatten(-2)
        posmat = posmat.narrow(-1, 1, n_out * (n_in + n_neg - 1))

        # example:
        #p0[-2], p0[-1], | p0[0], p0[1], p0[2], p0[3] |,
        #p1[-3], p1[-2]  | p1[-1], p1[0], p1[1], p1[2] |,
        #p1[3], p2[-3],  | p2[-2], p2[-1], p2[0], p2[1]|,
        #p2[2], p2[3] ,  |p3[-3], p3[-2], p3[-1], p3[0],|

        posmat = posmat.view(*posmat.shape[:-1], n_out, n_in + n_neg - 1)
        return posmat[..., n_neg-1 : ]

    def train(self, mode: bool = True):
        self.sel_hist_dst = []
        return super().train(mode)

    def get_loss_on_hist(self, l: List[torch.Tensor]) -> torch.Tensor:
        assert l[0].ndim == 3
        l = [t.flatten(end_dim=-2) for t in l]
        sel = torch.cat(l, -2)
        sel_d = F.log_softmax(sel, dim=-1)
        sel_d = framework.utils.distributed_ops.log_mean(sel_d, -2)
        return self.perplexity_reg * ( - utils.entropy_l(sel_d).mean())

    def get_reg_loss(self) -> Dict[str, torch.Tensor]:
        l = super().get_reg_loss()
        if self.sel_hist_dst:
            l["moe_att_entropy_dst"] = self.get_loss_on_hist(self.sel_hist_dst) if self.mode == "my" else sum(self.sel_hist_dst)
            self.sel_hist_dst = []

        return l

    def gate_topk(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(x) < self.expert_dropout
                x2 = x.masked_fill(mask, float('-inf'))
            else:
                x2 = x
            _, sel_index = x2.topk(self.n_heads, dim=-1, sorted=False)

        y = torch.gather(x, -1, sel_index)
        return y, sel_index

    def get_sel_my(self, t: torch.Tensor, w: torch.Tensor):
        sel = F.linear(t, w)
        sel_val, sel_index = self.gate_topk(sel)

        if self.selection_mode == "softmax":
            sel_val = sel_val.softmax(-1)
        elif self.selection_mode == "sigmoid":
            sel_val = sel_val.sigmoid()
        else:
            raise ValueError("Unknown selection mode: " + self.selection_mode)

        if self.training and self.perplexity_reg > 0:
            self.sel_hist_dst.append(sel)

        sel_index_pp = [cvmm_prepare_sel(sel_index[..., h].int(), self.n_experts) for h in range(self.n_heads)]
        return sel_val, sel_index, sel_index_pp

    def get_sel_moa(self, t: torch.Tensor, w: torch.Tensor):
        logits = F.linear(t, w)
        probs = logits.softmax(-1)
        top_k_gates, sel_index = self.gate_topk(probs)

        if self.training:
            if self.cvloss > 0 or self.switchloss > 0:
                zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype)
                gates = zeros.scatter(1, sel_index, top_k_gates)
                gates = gates.flatten(end_dim=-2)
                counts = (gates > 0).float().sum(0)

            loss = 0
            if self.cvloss > 0:
                loss += self.cvloss * self.compute_cvloss(gates)
            if self.switchloss > 0:
                loss += self.switchloss * self.compute_switchloss(probs.flatten(end_dim=-2), counts)
            if self.zloss > 0:
                loss += self.zloss * self.compute_zloss(logits.flatten(end_dim=-2))

            self.sel_hist_dst.append(loss)

        sel_index_pp = [cvmm_prepare_sel(sel_index[..., h].int(), self.n_experts) for h in range(self.n_heads)]
        return top_k_gates, sel_index, sel_index_pp


    def get_sel(self, t: torch.Tensor, w: torch.Tensor):
        if self.mode == "my":
            return self.get_sel_my(t, w)
        elif self.mode == "moa":
            return self.get_sel_moa(t, w)
        else:
            raise ValueError("Unknown mode: " + self.mode)

    def before_loss(self):
        self.iter += 1
        if self.iter % 100 == 0:
            sorted_counts = self.sel_counts_dst_100.sort(descending=True).values
            self.log("sel_counts/dst", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)

            self.sel_counts_dst_100 = 0

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                pos_offset: Optional[int] = None, need_weights: bool = False):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]

        if pos_offset is None:
            assert curr_state.shape[1] == attend_to.shape[1], "If attend_to has different shape than curr_state, pos_offset should be provided"
            pos_offset = 0

        dst_sel_val, raw_dst_sel_index, dst_sel_index = self.get_sel(curr_state, self.sel_dst)

        if self.training and self.iter % 10 == 0:
            self.sel_counts_dst_100 += F.one_hot(raw_dst_sel_index.flatten(), self.n_experts).sum(0)

        scale = self.scale.sqrt()

        pemb = self.get_pos_subset(attend_to.shape[-2], pos_offset)
        k_pos = F.linear(pemb, self.pos_to_pk) * scale

        k, v = (attend_to @ self.data_to_kv).split(self.projection_size, dim=-1)
        k = k * scale

        total_res = []
        for ah in range(self.n_heads):
            q = cvmm(curr_state, dst_sel_index[ah], self.data_to_q) * scale

            qc = qp = q

            kd = self.dropout(k)

            att = self.shift(qp @ self.dropout(k_pos).transpose(-2,-1)) + qc @ kd.transpose(-2, -1)
            att.masked_fill_(self.get_mask_tensor(attend_to.shape[-2], mask), float('-inf'))
            att = F.softmax(att, dim=-1)
            res = att @ v

            total_res.append(cvmm(res, dst_sel_index[ah], self.out_proj) * dst_sel_val[..., ah : ah + 1])

        return sum(total_res)
