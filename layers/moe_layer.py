import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, List, Union, Optional
from layers import LoggingLayer
from layers import RegularizedLayer
from framework import utils
import framework
import math
from layers import OncePerIterLayer
from layers.cvmm import CVMMSel, cvmm_prepare_sel2, cvmm
import os


class MoE(LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, dmodel: int, n_experts: int, expert_size: int, n_heads: int,
                 dropout: float = 0, weight_scale: float = 1.0,
                 dropout_mode: str = "none", selection_mode: str = "sigmoid", perplexity_reg: float = 0.0,
                 norm_keys: bool = False,
                 perplexity_reg_mode: str="step", n_random: int = 0, reg_type: str = "entropy",
                 topk_mode: str = "full", activation_after_topk: bool = False,
                 activation = lambda x: F.relu(x, inplace=True),
                 normalize_expert_sel_init: bool = False, norm_key_init: bool = False, norm_value_init: bool = False,
                 identical_init: bool = False,
                 rescale_normed: bool = False, sel_norm: str = "none",
                 v_dim: Optional[int] = None,
                 expert_dropout: float = 0.0,
                 sync_distributed: bool = False,
                 modulation_amplitude: float = 0.5,
                 ppl_past_blocks: int = 0):

        super().__init__()
        self.k_dim = dmodel
        self.v_dim = v_dim if v_dim is not None else dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.dropout = dropout
        self.dropout_mode = dropout_mode
        self.selection_mode = selection_mode
        self.perplexity_reg = perplexity_reg
        self.k_vec_dim = self.k_dim
        self.n_heads = n_heads
        self.norm_keys = norm_keys
        self.perplexity_reg_mode = perplexity_reg_mode
        self.n_random = n_random
        self.reg_type = reg_type
        self.topk_mode = topk_mode
        self.activation_after_topk = activation_after_topk
        self.activation = activation
        self.weight_scale = weight_scale
        self.normalize_expert_sel_init = normalize_expert_sel_init
        self.norm_key_init = norm_key_init
        self.norm_value_init = norm_value_init
        self.identical_init = identical_init
        self.layer = 0
        self.initalized = False
        self.rescale_normed = rescale_normed
        self.sel_norm = sel_norm
        self.was_training = True
        self.expert_dropout = expert_dropout
        self.reg_counts = 0
        self.sync_distributed = sync_distributed and torch.distributed.is_initialized()
        self.modulation_amplitude = modulation_amplitude
        self.record_all_expert_sel_counts = False
        self.ppl_past_blocks = ppl_past_blocks
        self.blocks_for_ppl = []
        self.recorded_inputs = []

        self.coocurence = None

        assert self.selection_mode in {"gate", "sigmoid", "sinkhorn", "sinkhorn2", "sinkmoid", "sinkmax", "sinkhorn_local", "mul", "sinkmoid2", "sinkmax2"}
        assert self.perplexity_reg_mode in {"step", "global", "time", "global_time"}
        assert self.dropout_mode in {"none", "score"}
        assert self.reg_type in {"perplexity", "variance", "entropy", "l2", "switch"}
        assert self.topk_mode in {"full", "l1_approx", "approx"}
        assert self.sel_norm in {"none", "cos", "input", "weights"}

        self.register_buffer("iter", torch.tensor(0, dtype=torch.int64), persistent=False)

        if selection_mode in {"mul"} and activation_after_topk:
            raise ValueError("Activation after topk is not supported with mul selection")

        self.keys = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim, self.expert_size))

        self.values = torch.nn.Parameter(torch.empty(self.n_experts, self.expert_size, self.v_dim))

        self.expert_sel = torch.nn.Parameter(torch.empty(self.n_experts, self.k_vec_dim))
        self.sel = lambda x: F.linear(x, self.expert_sel)

        torch.nn.init.normal_(self.expert_sel, std=self.k_vec_dim ** -0.5 * weight_scale)
        torch.nn.init.normal_(self.keys, std=dmodel ** -0.5 * weight_scale)
        torch.nn.init.normal_(self.values, std=self.size ** -0.5 * weight_scale)
        self.sel_hist = []
        self.index_sel_counts = 0
        self.index_sel_norm = 0

        self.index_sel_counts_100 = 0
        self.index_sel_norm_100 = 0

        self.sel_count_log = None

        self.all_expert_sel_counts = []
        self.all_expert_sel_soft = []

        self.register_buffer("kv_sel_counts", torch.zeros(self.n_experts, self.expert_size), persistent=False)
        self.register_buffer("kv_sel_counts_100", torch.zeros_like(self.kv_sel_counts))

        if self.rescale_normed and self.sel_norm != "none":
            self.sel_scale = torch.nn.Parameter(torch.ones([1]))
        else:
            self.sel_scale = 1.0

        self.register_buffer("seq", torch.arange(max(self.n_heads, self.n_experts, self.k_dim, self.v_dim), dtype=torch.long), persistent=False)
        self.regroup_weights()

        if self.ppl_past_blocks > 0 and self.reg_type not in {"perplexity", "entropy"}:
            print(f"Warning: ppl_past_blocks>0 (currently {self.ppl_past_blocks}) is only supported with perplexity and entropy regularization")

    def keys_to_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        k = keys.view(self.n_experts, self.k_vec_dim, self.expert_size)
        return k.permute(0, 2, 1).contiguous().view(-1, self.k_vec_dim)

    def keys_from_logical_order(self, keys: torch.Tensor) -> torch.Tensor:
        return keys.view(self.n_experts, self.expert_size, self.k_vec_dim).permute(0, 2, 1).contiguous().view(self.n_experts * self.k_vec_dim, self.expert_size)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def regroup_weights(self) -> Optional[torch.Tensor]:
        with torch.no_grad():
            if self.norm_key_init:
                self.renorm_keep_std(self.keys.view(self.n_experts, self.k_vec_dim, self.expert_size), dim=1)

            if self.norm_value_init:
                self.renorm_keep_std(self.values, dim=1)

            if self.identical_init:
                k = self.keys.view(self.n_experts, self.k_vec_dim, self.expert_size)
                self.keys.set_(k[:1].expand_as(k).reshape_as(self.keys))

                v = self.values.view(self.n_experts, self.expert_size, self.v_dim)
                self.values.set_(v[:1].expand_as(v).reshape_as(self.values))

            if self.normalize_expert_sel_init:
                self.renorm_keep_std(self.expert_sel, dim=1)

    def ani(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        chunk_size = 32

        xnorm = F.normalize(x, 2, dim=-1)

        accu = 0
        for i in range(0, x.shape[0], chunk_size):
            a = xnorm[i: i + chunk_size]
            sims = xnorm @ a.T
            sims[i : i + chunk_size].fill_diagonal_(0)
            accu += sims.sum()

        return accu / (x.shape[0] * (x.shape[0] - 1))

    def log_expert_sel_usage(self, prefix: str, channel_sel_counts: torch.Tensor):
        sel_nonzero = (channel_sel_counts != 0).type(torch.float).sum(axis=-1) / self.expert_size
        self.log(f"{prefix}/mean", sel_nonzero.mean())
        self.log(f"{prefix}/min", sel_nonzero.min())
        self.log(f"{prefix}/max", sel_nonzero.max())


    def pre_train_forward(self):
        if self.norm_keys:
            with torch.no_grad():
                self.keys.div_(self.keys.norm(dim=-1, keepdim=True))

        if self.training and not self.was_training:
            sorted_counts = self.index_sel_counts.sort(descending=True).values
            self.log("test_exert_channel_usage", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)

        self.layer = 0
        if self.sel_hist:
            self.sel_hist = []
        self.index_sel_counts = 0
        self.index_sel_norm = 0
        self.reg_counts = 0

    def before_loss(self):
        if self.sel_hist:
            # Concatenate against time dimension. Important for the within-batch regularization
            sel = torch.cat(self.sel_hist, -2)
            self.add_perplexity_reg(sel)

            self.sel_hist = []

        if self.index_sel_norm > 0:
            if self.training:
                with torch.no_grad():
                    self.log("usag_rel_perplexity_all_layers", utils.relative_perplexity(self.index_sel_counts / self.index_sel_norm))
                    self.log("dead_expert_proportion_all_layers", (self.index_sel_counts == 0).float().sum() / self.n_experts)

                    self.log_expert_sel_usage("exert_channel_usage", self.kv_sel_counts)

                    self.kv_sel_counts_100.add_(self.kv_sel_counts)
                    self.kv_sel_counts.zero_()

                    self.index_sel_counts_100 = self.index_sel_counts_100 + self.index_sel_counts
                    self.index_sel_norm_100 = self.index_sel_norm_100 + self.index_sel_norm

                    if self.training and self.iter % 100 == 0:
                        norm_cnt = self.index_sel_counts_100 / self.index_sel_norm_100
                        self.log("usag_rel_perplexity_100", utils.relative_perplexity(norm_cnt))
                        self.log("dead_expert_proportion_100", (self.index_sel_counts_100 == 0).float().sum() / self.n_experts)

                        sorted_counts = self.index_sel_counts_100.sort(descending=True).values
                        self.log("usage_counts_100", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)


                        self.log_expert_sel_usage("exert_channel_usage_100", self.kv_sel_counts_100)
                        self.kv_sel_counts_100.zero_()

                        self.index_sel_counts_100 = 0
                        self.index_sel_norm_100 = 0

                        self.log("ani/keys", self.ani(self.keys_to_logical_order(self.keys)))
                        self.log("ani/values", self.ani(self.values.flatten(0, -2)))
                        self.log("ani/expert_sel", self.ani(self.expert_sel.T))

        if self.training:
            self.iter += 1

    def topk(self, x: torch.Tensor, k: int, approx: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if approx:
            x = x.view(*x.shape[:-1], k, -1)
            scores, ind = x.max(-1)
            return scores, self.seq[:k] * x.shape[-1] + ind
        else:
            return x.topk(k, dim=-1, sorted=False)

    def rolling_logsumexp(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate calculating logsumexp over a bigger batch than the current one. Will have stale values, but that
        # should not matter much later in training.
        if self.ppl_past_blocks == 0 or not self.training:
            return F.log_softmax(x, dim=-1)
        else:
            if len(self.blocks_for_ppl) == self.ppl_past_blocks:
                self.blocks_for_ppl.pop(0)

            self.blocks_for_ppl.append(x)
            res = F.log_softmax(torch.cat(self.blocks_for_ppl, dim=0), dim=-1)
            self.blocks_for_ppl[-1] = self.blocks_for_ppl[-1].detach()
            return res

    def add_perplexity_reg(self, sel: torch.Tensor):
        sync_distributed = self.sync_distributed and (self.perplexity_reg_mode not in {"time", "global_time"})

        if self.perplexity_reg_mode in {"time", "global_time"}:
            sel = sel.flatten(0, -3)
        else:
            sel = sel.flatten(0, -2)

        # Note: sel are raw logits, no matter what activation is used
        if self.perplexity_reg > 0:
            if self.reg_type == "perplexity":
                sel_d = self.rolling_logsumexp(sel)
                sel_d = framework.utils.distributed_ops.log_mean(sel_d, -2, self.sync_distributed)
                loss = lambda: self.perplexity_reg * ( - utils.relative_perplexity_l(sel_d).mean())
            elif self.reg_type == "entropy":
                sel_d = self.rolling_logsumexp(sel)
                sel_d = framework.utils.distributed_ops.log_mean(sel_d, -2, self.sync_distributed)
                loss = lambda: self.perplexity_reg * ( - utils.entropy_l(sel_d).mean())
            elif self.reg_type == "variance":
                if sync_distributed:
                    raise NotImplementedError("Variance regularization is not supported in distributed mode")
                avg_sel = sel.mean(-2)
                loss = lambda: self.perplexity_reg * avg_sel.var(-1).mean()
            elif self.reg_type == "l2":
                loss = lambda: self.perplexity_reg * sel.pow(2).mean()
            elif self.reg_type == "switch":
                if sync_distributed:
                    torch.distributed.all_reduce(self.reg_counts, op=torch.distributed.ReduceOp.SUM)

                p_sel_real = self.reg_counts / self.reg_counts.sum(-1, keepdims=True)
                if self.perplexity_reg_mode in {"time", "global_time"}:
                    p_sel_real = p_sel_real.unsqueeze(-2)

                loss = lambda: self.perplexity_reg * (F.softmax(sel, dim=-1) * p_sel_real).mean()
                self.reg_counts = 0
            else:
                assert False

            self.add_reg(loss, "moe")

    def compute_scores(self, input: torch.Tensor, index: CVMMSel, expert_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = cvmm(input, index, self.keys)

        if self.selection_mode in {"mul"}:
            scores = scores * expert_scores[..., None]
        elif self.selection_mode in {"gate", "sigmoid", "sinkhorn", "sinkhorn2", "sinkmoid", "sinkmax", "sinkmoid2"}:
            # Handle it later
            pass

        scores = self.activation(scores)

        plot_training = self.train and self.iter % 10 == 0
        if plot_training:
            with torch.no_grad():
                gt0 = (scores > 0).float()
                gt0_s = gt0.sum()

                if plot_training:
                    self.log("relu_pass_rate", gt0_s / scores.numel())

                    self.kv_sel_counts.index_add_(0, index.raw_sel.flatten(), gt0.flatten(end_dim=-2))

        if self.dropout > 0 and self.dropout_mode != "none":
            scores = F.dropout(scores, self.dropout, training=self.training)

        return scores

    def sel_activation(self, sel: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_sel = sel
        if self.selection_mode in {"sigmoid"}:
            sel = torch.sigmoid(sel)
        elif self.selection_mode in {"mul"}:
            sel = sel.abs()
            reg_sel = sel
        elif self.selection_mode in {"gate"}:
            sel = F.softmax(sel, dim=-1)
            with torch.no_grad():
                self.log("expert_rel_perplexity_per_selection", utils.relative_perplexity(sel).mean())
        else:
            assert False

        return sel, reg_sel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = 0

        in1 = in2 = input

        sel = self.sel(in1)
        # sel=sel.float()

        if self.sel_norm == "cos":
            sel = sel / (in1.norm(dim=-1, keepdim=True) * self.expert_sel.norm(dim=-1)[None]) * self.sel_scale
        elif self.sel_norm == "weights":
            sel = sel * (self.sel_scale / self.expert_sel.norm(dim=-1)[None])
        elif self.sel_norm == "input":
            sel = sel * (self.sel_scale / in1.norm(dim=-1, keepdim=True))

        sel_raw = reg_sel = sel

        inv_val = float("-inf")

        if not self.activation_after_topk:
            # Sinkhorn should be always applied before top-k
            sel, reg_sel = self.sel_activation(sel, input.shape[-2])
            inv_val = 0

        if self.training and self.expert_dropout > 0:
            if self.selection_mode not in {"sigmoid", "gate"}:
                raise ValueError("Expert dropout not supported in this mode")

            mask = torch.rand_like(sel) < self.expert_dropout
            sel2 = sel.masked_fill(mask, inv_val)
        else:
            sel2 = sel

        sel_val, sel_index = self.topk(sel2, self.n_heads, self.topk_mode in {"l1_approx", "approx"})

        if self.activation_after_topk or (self.selection_mode in {"mul"}):
            sel_val = torch.gather(sel_raw, -1, sel_index)
            sel_val, reg_sel = self.sel_activation(sel_val, input.shape[-2])


        record_counts_now = (self.training and self.iter % 10 == 0) or (not self.training) or (self.record_all_expert_sel_counts)

        if not self.training:
            sel_index_flat = sel_index.flatten(end_dim=-2)
            if self.coocurence is None:
                self.coocurence = torch.zeros([self.n_experts, self.n_experts], device=sel_index_flat.device, dtype=torch.long)

            for h1 in range(self.n_heads):
                for h2 in range(self.n_heads):
                    ind_flat = sel_index_flat[..., h1] * self.n_experts + sel_index_flat[..., h2]
                    values = torch.tensor([1], device=self.coocurence.device, dtype=self.coocurence.dtype).expand_as(ind_flat)
                    # values = sel_val[..., h2].flatten()
                    self.coocurence.flatten().put_(ind_flat, values, accumulate=True)
                    # self.coocurence[sel_index_flat[..., h1], sel_index_flat[..., h2]] += 1

        if record_counts_now or self.reg_type == "switch":
            reg_counts = F.one_hot(sel_index, self.n_experts).type_as(input)

        if self.reg_type == "switch":
            reg_counts2 = reg_counts.view(*input.shape[:-2], input.shape[-2] * self.n_heads, self.n_experts)
            if self.perplexity_reg_mode == "time":
                reg_counts2 = reg_counts2.sum(-2)
            else:
                reg_counts2 = reg_counts2.flatten(end_dim=-2).sum(0)

            self.reg_counts = self.reg_counts + reg_counts2

        if record_counts_now:
            with torch.no_grad():
                sel_counts = reg_counts.flatten(end_dim=-2).sum(0)
                cnt = sel_index.nelement()

                p_expert_sel = sel_counts / cnt

                self.index_sel_counts = self.index_sel_counts + sel_counts
                self.index_sel_norm = self.index_sel_norm + cnt

                if self.record_all_expert_sel_counts:
                    softcnt = torch.zeros_like(sel_counts, dtype=sel_val.dtype)
                    softcnt.index_add_(0, sel_index.flatten(), sel_val.flatten())

                    self.all_expert_sel_soft.append(softcnt)
                    self.all_expert_sel_counts.append(sel_counts)

                if self.training:
                    self.log("min_sel_score", sel_val.min(dim=-1).values.mean())
                    self.log("max_sel_score", sel_val.max(dim=-1).values.mean())

                    sel_oh = F.one_hot(sel_index, self.n_experts).sum(-2).bool()
                    if self.layer >= 1 and self.training:
                        self.log(f"layer_sel_overlap_{self.layer}", ((self.prev_sel_oh & sel_oh).sum(-1).float() / self.n_heads).mean())

                    self.prev_sel_oh = sel_oh

                    ppl = utils.relative_perplexity(p_expert_sel)
                    self.log("usage_rel_perplexity", ppl)
                    self.log("dead_expert_proportion", (p_expert_sel == 0).float().sum() / self.n_experts)

        if self.perplexity_reg_mode in {"step", "time"}:
            self.add_perplexity_reg(reg_sel)
        elif self.perplexity_reg > 0 and self.training:
            self.sel_hist.append(reg_sel)

        sel_indices = cvmm_prepare_sel2(sel_index.int())

        scores = self.compute_scores(in2, sel_indices, sel_val)

        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = sel_val
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None

        if self.selection_mode not in {"gate", "sigmoid"}:
            sel_indices.reduction_weight = torch.ones_like(sel_indices.reduction_weight)

        out = cvmm(scores, sel_indices, self.values)

        self.layer += 1

        self.was_training = self.training
        res = out.view(*input.shape[:-1], self.v_dim)
        return res

    def dump_logs(self, save_dir: str):
        if self.coocurence is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.coocurence, os.path.join(save_dir, "coocurence.pt"))

    def get_logs(self) -> Dict[str, Any]:
        res = super().get_logs()

        if self.coocurence is not None:
            coo = self.coocurence / self.coocurence.diagonal().clamp(min=1)[:, None]
            res["expert_coocurence"] = framework.visualize.plot.Heatmap(coo, xlabel="expert", ylabel="expert", textval=False)
            self.coocurence = None
        return res
