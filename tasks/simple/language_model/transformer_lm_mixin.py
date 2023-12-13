import framework
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import math
from typing import List, Tuple, Dict, Any
from models import TransformerLanguageModel
from ... import task, args
from layers.transformer import RelativeTransformerEncoderLayer, PrelnRelativeTransformerEncoderLayer
from layers.transformer.relative_moe_transformer import RelativeMoeTransformerEncoderLayer
from layers.transformer.fast_rope_transformer import FastRopeTransformerEncoderLayer
from layers.transformer.moe_attention_relative_transformer import MoeAttentionRelativeTransformerEncoderLayer
from layers.moe_layer import MoE
from interfaces import Result
from layers import LayerVisualizer

from layers.transformer.full_moe_relative_attention import FullMoeRelativeAttentionCore

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-lm.trafo.context_blocks", default=1)
    parser.add_argument("-lm.trafo.test_context_blocks", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.trafo.test_pos_clamp", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.trafo.same_length_eval", default=False)
    parser.add_argument("-lm.trafo.same_length", default=False)
    parser.add_argument("-lm.trafo.last_layer_context", default=False)
    parser.add_argument("-lm.trafo.xl_init", default=False)
    parser.add_argument("-lm.trafo.embedding_mode_init", default="default", choice=["default", "scale_to_sqrt_dmodel", "init_to_sqrt_dmodel", "one_and_scale_to_sqrt_dmodel", "like_preln"])
    parser.add_argument("-pkm.n_heads", default=1)
    parser.add_argument("-moe.n_experts", default=128)
    parser.add_argument("-moe.expert_size", default=128)
    parser.add_argument("-moe.selection_mode", default="sigmoid", choice=["gate", "sigmoid",  "mul"])
    parser.add_argument("-moe.perplexity_reg", default=0.0)
    parser.add_argument("-moe.perplexity_reg_mode", default="step", choice=["step", "global", "time", "global_time"])
    parser.add_argument("-moe.reg_type", default="entropy", choice=["perplexity", "variance", "entropy", "l2", "switch", "normal"])
    parser.add_argument("-moe.norm_keys", default=False)
    parser.add_argument("-moe.n_random", default=0)
    parser.add_argument("-moe.topk_mode", default="full", choice=["full", "l1_approx", "approx"])
    parser.add_argument("-moe.activation_after_topk", default=False)
    parser.add_argument("-moe.drop_parallel", default=True)
    parser.add_argument("-moe.norm_key_init", default=False)
    parser.add_argument("-moe.norm_value_init", default=False)
    parser.add_argument("-moe.identical_init", default=False)
    parser.add_argument("-moe.sel_lr_multipler", default=1.0)
    parser.add_argument("-moe.expert_lr_multipler", default=1.0)
    parser.add_argument("-moe.sel_norm", default="none", choice=["none", "cos", "input", "weights"])
    parser.add_argument("-moe.dropout_factor", default=1.0)
    parser.add_argument("-moe.drop_expert", default=0.0)
    parser.add_argument("-moe.sync_distributed", default=True)
    parser.add_argument("-moe.modulation_amplitude", default=0.5)
    parser.add_argument("-moe.init_scale", default=1.0)
    parser.add_argument("-moe.norm_expert_sel_init", default=False)
    parser.add_argument("-kvmem.dropout", default="none", choice=["none", "early", "late", "weight", "score"])
    parser.add_argument("-kvmem.norm_values", default=False)
    parser.add_argument("-transformer.topk_value", default=32)
    parser.add_argument("-transformer.activation", default="relu", choice=["relu", "topk", "gelu", "identity", "sigmoid", "softmax"])
    parser.add_argument("-transformer.p_drop_layer", default=0.0)
    parser.add_argument("-transformer.head_projection_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.ln_affine", default=True)
    parser.add_argument("-transformer.ln_after_attention", default=True)
    parser.add_argument("-moe.att.n_experts", default=4)
    parser.add_argument("-moe.att.variant", default="moa", choice=["moa", "simple", "qside", "full", "full_rope", "seq", "target"])
    parser.add_argument("-moe.att.enable", default=False)
    parser.add_argument("-moe.att.q_expert", default=True)
    parser.add_argument("-moe.att.k_expert", default=True)
    parser.add_argument("-moe.att.v_expert", default=True)
    parser.add_argument("-moe.att.o_expert", default=True)
    parser.add_argument("-moe.att.k", default=2)
    parser.add_argument("-moe.att.norm_qk", default=False)
    parser.add_argument("-moe.att.v_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-moe.att.same_sel", default=False)
    parser.add_argument("-moe.att.expert_dropout", default="none", parser=parser.float_or_none_parser)
    parser.add_argument("-moe.att.selection_mode", default="sigmoid", choice=["sigmoid", "softmax"])
    parser.add_argument("-moe.att.perplexity_reg", default="none", parser=parser.float_or_none_parser)
    parser.add_argument("-moe.att.qside_n_experts", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-moe.att.k", default=2)
    parser.add_argument("-moe.att.norm_ret", default=False)
    parser.add_argument("-moe.att.shared_experts", default=False)
    parser.add_argument("-moe.att.drop_expert", default="none", parser=parser.float_or_none_parser)
    parser.add_argument("-moe.att.kq_n_experts", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-moe.att.separate_kq_sel", default=False)
    parser.add_argument("-moe.att.norm_init", default=False)
    parser.add_argument("-rope.rotate_fraction", default=0.5)
    parser.add_argument("-rope.base", default=10000.0)
    parser.add_argument("-moa.mode", default="my", choice=["my", "moa"])
    parser.add_argument("-moa.cvloss", default=0.0)
    parser.add_argument("-moa.switchloss", default=0.0)
    parser.add_argument("-moa.zloss", default=0.0)
    parser.add_argument("-debug_plot_interval", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.plot_head_details", default=False)
    parser.add_argument("-plot.n_steps", default=-128)

@task()
class TransformerLMMixin:
    helper: framework.helpers.TrainingHelper

    def is_preln(self) -> bool:
        return "preln" in self.helper.args.transformer.variant

    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        nx = -x
        return torch.masked_fill(x, nx <= nx.kthvalue(self.helper.args.transformer.topk_value, keepdim=True)[0], 0)

    def get_layers(self) -> List[torch.nn.Module]:
        # pyright: reportOptionalMemberAccess=false
        if self.helper.args.transformer.activation == "relu":
            activation = F.relu
        elif self.helper.args.transformer.activation == "topk":
            activation = self.topk_activation
        elif self.helper.args.transformer.activation == "identity":
            activation = lambda x: x
        elif self.helper.args.transformer.activation == "sigmoid":
            activation = torch.sigmoid
        elif self.helper.args.transformer.activation == "gelu":
            activation = F.gelu
        elif self.helper.args.transformer.activation == "softmax":
            activation = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Invalid activation: {self.helper.args.transformer.activation}")

        base_args = dict(
            d_model=self.helper.args.state_size,
            nhead=self.helper.args.transformer.n_heads,
            dropout=self.helper.args.dropout,
            activation=activation
        )

        if self.helper.args.transformer.variant not in {"preln_moe", "moe"}:
            base_args["dim_feedforward"]=int(self.helper.args.state_size * self.helper.args.transformer.ff_multiplier)


        extra_args = {} if not self.helper.args.transformer.variant.endswith("_gelu") else {
            "activation": F.gelu,
            "drop_expand": False
        }


        if self.helper.args.transformer.variant in {"preln_relative"}:
            mklayer = lambda: PrelnRelativeTransformerEncoderLayer(
                **base_args, **extra_args, test_pos_clamp=self.helper.args.lm.trafo.test_pos_clamp,
                n_layers=self.helper.args.transformer.encoder_n_layers,
                head_projection_size=self.helper.args.transformer.head_projection_size,)
        elif self.helper.args.transformer.variant in {"preln_moeatt"}:
            mklayer = lambda: MoeAttentionRelativeTransformerEncoderLayer(
                **base_args, **extra_args, moe_att_n_experts=self.helper.args.moe.att.n_experts,
                n_layers=self.helper.args.transformer.encoder_n_layers,
                head_projection_size=self.helper.args.transformer.head_projection_size,
                att_perplexity_reg=self.helper.args.moe.perplexity_reg if self.helper.args.moe.att.perplexity_reg is None else self.helper.args.moe.att.perplexity_reg,
                expert_dropout=self.helper.args.moe.drop_expert if self.helper.args.moe.att.drop_expert is None else self.helper.args.moe.att.drop_expert,
                att_selection_mode=self.helper.args.moe.att.selection_mode,
                preln=self.is_preln(),
                attention_variant=self.helper.args.moe.att.variant,
                q_expert=self.helper.args.moe.att.q_expert,
                k_expert=self.helper.args.moe.att.k_expert,
                v_expert=self.helper.args.moe.att.v_expert,
                o_expert=self.helper.args.moe.att.o_expert,
                norm_qk_score=self.helper.args.moe.att.norm_qk,
                v_projection_size=self.helper.args.moe.att.v_size,
                same_sel=self.helper.args.moe.att.same_sel,
                moe_k=self.helper.args.moe.att.k,
                qside_n_experts=self.helper.args.moe.att.qside_n_experts,
                shared_experts=self.helper.args.moe.att.shared_experts,
                kq_n_experts=self.helper.args.moe.att.kq_n_experts,
                separate_kq_sel=self.helper.args.moe.att.separate_kq_sel,
                moa_mode=self.helper.args.moa.mode,
                cvloss=self.helper.args.moa.cvloss,
                switchloss=self.helper.args.moa.switchloss,
                zloss=self.helper.args.moa.zloss,
                rotate_fraction=self.helper.args.rope.rotate_fraction,
                rope_base=self.helper.args.rope.base,
                moeatt_norm_init=self.helper.args.moe.att.norm_init)
        elif self.helper.args.transformer.variant in {"preln_rope", "rope"}:
            mklayer = lambda: FastRopeTransformerEncoderLayer(
                **base_args, **extra_args,
                n_layers=self.helper.args.transformer.encoder_n_layers,
                head_projection_size=self.helper.args.transformer.head_projection_size,
                preln=self.is_preln(), rotate_fraction = self.helper.args.rope.rotate_fraction,
                rope_base=self.helper.args.rope.base)
        elif self.helper.args.transformer.variant in {"preln_moe", "moe"}:
            # def __init__(self, d_model, nhead, n_bins: int, bin_size: int, n_layers: int, dim_feedforward=2048,
            mklayer = lambda: RelativeMoeTransformerEncoderLayer(
                **base_args, **extra_args, preln=self.is_preln(),
                test_pos_clamp=self.helper.args.lm.trafo.test_pos_clamp,
                n_layers=self.helper.args.transformer.encoder_n_layers,
                n_experts=self.helper.args.moe.n_experts,
                expert_size=self.helper.args.moe.expert_size,
                dropout_mode=self.helper.args.kvmem.dropout,
                selection_mode=self.helper.args.moe.selection_mode,
                perplexity_reg=self.helper.args.moe.perplexity_reg,
                n_heads=self.helper.args.pkm.n_heads,
                norm_keys=self.helper.args.moe.norm_keys,
                perplexity_reg_mode=self.helper.args.moe.perplexity_reg_mode,
                n_random=self.helper.args.moe.n_random,
                reg_type=self.helper.args.moe.reg_type,
                topk_mode=self.helper.args.moe.topk_mode,
                head_projection_size=self.helper.args.transformer.head_projection_size,
                activation_after_topk=self.helper.args.moe.activation_after_topk,
                drop_parallel=self.helper.args.moe.drop_parallel,
                norm_key_init=self.helper.args.moe.norm_key_init,
                norm_value_init=self.helper.args.moe.norm_value_init,
                normalize_expert_sel_init=self.helper.args.moe.norm_expert_sel_init,
                identical_init=self.helper.args.moe.identical_init,
                sel_norm=self.helper.args.moe.sel_norm,
                ln_affine=self.helper.args.transformer.ln_affine,
                moe_dropout_factor=self.helper.args.moe.dropout_factor,
                drop_expert=self.helper.args.moe.drop_expert,
                sync_distributed=self.helper.args.moe.sync_distributed,
                modulation_amplitude=self.helper.args.moe.modulation_amplitude,
                moe_init_scale=self.helper.args.moe.init_scale,
                moe_attention=self.helper.args.moe.att.enable,
                moe_att_n_experts=self.helper.args.moe.att.n_experts,
                moe_att_expert_dropout=self.helper.args.moe.drop_expert if self.helper.args.moe.att.drop_expert is None else self.helper.args.moe.att.drop_expert,
                moe_att_selection_mode=self.helper.args.moe.att.selection_mode,
                moe_att_variant=self.helper.args.moe.att.variant,
                moe_att_ppl_reg=self.helper.args.moe.perplexity_reg if self.helper.args.moe.att.perplexity_reg is None else self.helper.args.moe.att.perplexity_reg,
                moe_att_k=self.helper.args.moe.att.k,
                q_expert=self.helper.args.moe.att.q_expert,
                k_expert=self.helper.args.moe.att.k_expert,
                v_expert=self.helper.args.moe.att.v_expert,
                o_expert=self.helper.args.moe.att.o_expert,
                v_projection_size=self.helper.args.moe.att.v_size,
                qside_n_experts=self.helper.args.moe.att.qside_n_experts,
                moe_att_shared_experts=self.helper.args.moe.att.shared_experts,
                moe_att_kq_n_experts=self.helper.args.moe.att.kq_n_experts,
                moe_att_separate_kq_sel=self.helper.args.moe.att.separate_kq_sel,
                rotate_fraction=self.helper.args.rope.rotate_fraction,
                rope_base=self.helper.args.rope.base,
                moe_att_norm_init=self.helper.args.moe.att.norm_init,
                moe_att_same_sel=self.helper.args.moe.att.same_sel,
                moe_att_norm_retrieval=self.helper.args.moe.att.norm_ret)
        else:
            assert False, f"Invalid variant \"{self.helper.args.transformer.variant}\""

        layers = [mklayer() for _ in range(self.helper.args.transformer.encoder_n_layers)]
        return layers


    def fix_init(self, model):
        init_std = 0.02

        torch.nn.init.normal_(model.embedding.weight, 0.0, init_std)
        # torch.nn.init.normal_(model.embedding_adapter.weight, 0.0, init_std)

        initialized = 0
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)) and hasattr(m, "weight"):
                torch.nn.init.normal_(m.weight, 0.0, init_std)
                initialized += m.weight.numel()
            if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                initialized += m.bias.numel()
            if isinstance(m, (torch.nn.LayerNorm)) and m.weight is not None:
                torch.nn.init.normal_(m.weight, 1.0, init_std)
                initialized += m.weight.numel()
            if isinstance(m, MoE):
                torch.nn.init.normal_(m.keys, 0.0, init_std)
                torch.nn.init.normal_(m.values, 0.0, init_std)
                if m.expert_sel is not None:
                    torch.nn.init.normal_(m.expert_sel, 0.0, init_std)
                    m.renorm_keep_std(m.expert_sel)
                    initialized += m.expert_sel.numel()
                initialized += m.keys.numel() + m.values.numel()
            if isinstance(m, (FullMoeRelativeAttentionCore)):
                for p in m.parameters():
                    torch.nn.init.normal_(p, 0.0, init_std)
                    initialized += p.numel()

                for s in m.selections.values():
                    m.renorm_keep_std(s)

        print(f"Reinitialized {initialized/self.n_weights*100:.3f}% weights")

    def create_model(self) -> torch.nn.Module:
        self.validation_started_on = None
        # pyright: reportOptionalMemberAccess=false
        tlayers = self.get_layers()

        model = TransformerLanguageModel(
            len(self.train_set.vocabulary), self.helper.args.embedding_size,
            self.helper.args.state_size, self.helper.args.dropout,
            tied_embedding=self.helper.args.tied_embedding,
            layers=tlayers, n_prev_states=self.helper.args.lm.trafo.context_blocks,
            n_prev_states_test=self.helper.args.lm.trafo.test_context_blocks,
            same_length_eval=self.helper.args.lm.trafo.same_length_eval,
            p_drop_layer=self.helper.args.transformer.p_drop_layer,
            same_length=self.helper.args.lm.trafo.same_length,
            use_last_state=self.helper.args.lm.trafo.last_layer_context,
            norm_before_output=self.is_preln())

        self.n_weights = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            if self.is_preln():
                model.embedding_scale = 1.0
            elif self.helper.args.lm.trafo.xl_init:
                self.fix_init(model)
            elif self.helper.args.lm.trafo.embedding_mode_init=="scale_to_sqrt_dmodel":
                norm = model.embedding.weight.norm(dim=-1).mean()
                model.embedding_scale = math.sqrt(self.helper.args.state_size) / norm
            elif self.helper.args.lm.trafo.embedding_mode_init=="one_and_scale_to_sqrt_dmodel":
                norm = model.embedding.weight.norm(dim=-1).mean()
                model.embedding_scale = math.sqrt(self.helper.args.state_size)
                model.embedding.weight.mul_(1.0 / norm)
            elif self.helper.args.lm.trafo.embedding_mode_init=="init_to_sqrt_dmodel":
                norm = model.embedding.weight.norm(dim=-1, keepdim=True)
                model.embedding_scale=1.0
                model.embedding.weight.mul_(math.sqrt(self.helper.args.state_size) / norm)

        self.visualizer = LayerVisualizer(model, {
            "mha.plot_head_details": self.helper.args.transformer.plot_head_details,
            "mha.no_pos_vs_content": True
        })

        self.input_history = []
        return model


    def train_step(self) -> Tuple[Result, Dict[str, Any]]:
        if self.helper.args.kvmem.norm_values:
            with torch.no_grad():
                for m in self.model.modules():
                    if isinstance(m, torch.nn.EmbeddingBag):
                        m.weight.div_(m.weight.norm(dim=-1, keepdim=True))

        return super().train_step()

    def get_optimizer_param_list(self):
        params = list(self.model.parameters())
        sel_params = []
        expert_params = []

        if self.helper.args.moe.sel_lr_multipler != 1.0:
            for m in self.model.modules():
                if isinstance(m, MoE):
                    sel_params += [m.expert_sel]

        if self.helper.args.moe.expert_lr_multipler != 1.0:
            for m in self.model.modules():
                if isinstance(m, MoE):
                    expert_params += [m.keys, m.values]

        excluded_params = [id(p) for p in sel_params + expert_params]
        params = [p for p in params if id(p) not in excluded_params]

        if not excluded_params:
            return params

        return [
            {"params": params},
            {"params": sel_params, "lr": self.helper.args.lr * self.helper.args.moe.sel_lr_multipler},
            {"params": expert_params, "lr": self.helper.args.lr * self.helper.args.moe.expert_lr_multipler},
        ]

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        if (self.VIS_DATASET_FILTER is None) or (name in self.VIS_DATASET_FILTER):
            self.validation_started_on = name
            self.validation_step = 0

        return super().validate_on_name(name)

    def get_steplabels(self, data: Dict[str, torch.Tensor]) -> List[str]:
        out = self.train_set.vocabulary(data["data"][:, 0].cpu().numpy().tolist())
        inp = [self.train_set.vocabulary(x[:-1].cpu().numpy().tolist()) for x in self.input_history] + [out]
        return sum(inp, [])[:-1], out[1:]

    def run_model(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> Tuple[Result, Dict[str, Any]]:
        plot_now = ((ubatch == 0) and (self.helper.args.debug_plot_interval is not None) and \
                   ((self.helper.state.iter % self.helper.args.debug_plot_interval) == 0) and self.model.training)

        is_dumping = self.validation_started_on and self.helper.args.dump_validation_plots

        if plot_now or is_dumping:
            inp, outp = self.get_steplabels(data)
            params = {"steplabel": inp, "target_labels": outp}
            if self.helper.args.plot.n_steps:
                params["n_steps"] = self.helper.args.plot.n_steps

            self.visualizer.prepare(params)

        if ubatch == 0 and self.helper.args.lm.trafo.context_blocks > 0:
            if len(self.input_history) >= self.helper.args.lm.trafo.context_blocks:
                self.input_history.pop(0)
            self.input_history.append(data["data"][:, 0])

        res, plots = super().run_model(data, ubatch)

        if plot_now or is_dumping:
            plots.update({f"activations/{k}": v for k, v in self.visualizer.plot().items()})

        if is_dumping:
            os.makedirs(self.helper.args.dump_validation_plots, exist_ok=True)
            torch.save(plots, f"{self.helper.args.dump_validation_plots}/{self.validation_started_on}_{self.validation_step:04d}.pth")
            self.validation_step += 1

        return res, plots