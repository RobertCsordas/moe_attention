
from typing import Optional, Dict, List, Any, Union, Tuple
import torch
import torch.nn.functional as F
from .multi_head_attention import AttentionMask
import framework
import math
from ..cvmm import cvmm, cvmm_prepare_sel2, cvmm_prepare_sel, CVMMSel
from layers.regularized_layer import RegularizedLayer
from layers.once_per_iter_layer import OncePerIterLayer
from layers.logging_layer import LoggingLayer
import framework
from framework import utils
from collections import namedtuple
from layers.layer_with_visualization import LayerWithVisualization
from framework.visualize.plot import CustomPlot
import numpy as np
import wandb
from .rotary_pos_encoding import RotaryPosEncoding
import os

Selection = namedtuple('Selection', ['raw_sel', 'sel_val', 'raw_sel_index', 'sel_index'])

plt = None
make_axes_locatable = None
FigureCanvas = None
gridspec = None

def import_matplotlib():
    global plt
    global make_axes_locatable
    global FigureCanvas
    global gridspec
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.backends.backend_agg import FigureCanvas
    import matplotlib.gridspec as gridspec


class MoEAttentionPlot(CustomPlot):
    def __init__(self, map: Union[torch.Tensor],
                 x_sel : Dict[str, torch.Tensor],
                 y_sel : Dict[str, torch.Tensor],
                 xlabel: str, ylabel: str,
                 x_marks: Optional[List[str]] = None,
                 y_marks: Optional[List[str]] = None,
                 fps: float = 2, cmap = "auto", colorbar: bool = True, fontsize: float = 6):

        import_matplotlib()
        super().__init__()

        if torch.is_tensor(map):
            map = map.detach().cpu().numpy()

        self.map = map
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_marks = x_marks
        self.y_marks = y_marks
        self.fps = fps
        self.cmap = plt.cm.Blues if cmap=="auto" else cmap
        self.sel_cmap = self.cmap
        self.colorbar = colorbar
        self.font_size = fontsize
        self.shape_factor = 0.1
        self.x_sel = x_sel
        self.y_sel = y_sel

    def get_marks(self, m: Optional[Union[str, List[str]]], n: int):
        if m is None:
            return None

        assert len(m) == n
        return list(m)

    def to_video(self):
        import_matplotlib()

        data = self.map.astype(np.float32)

        x_marks = self.get_marks(self.x_marks, self.map.shape[2])
        y_marks = self.get_marks(self.y_marks, self.map.shape[1])

        xorder = list(sorted(self.x_sel.keys()))
        yorder = list(sorted(self.y_sel.keys()))

        x_sizes = [self.x_sel[k].shape[-1] for k in xorder]
        y_sizes = [self.y_sel[k].shape[-1] for k in yorder]

        n_rows = 1 + len(self.y_sel)
        n_cols = 1 + len(self.x_sel) + 1

        width_ratios = y_sizes + [data[0].shape[-1], 3]
        tot_width = sum(width_ratios)
        width_ratios = [s / tot_width for s in width_ratios]

        height_ratios = [data[0].shape[-2]] + x_sizes
        tot_height = sum(height_ratios)
        height_ratios = [s / tot_height for s in height_ratios]

        figure = plt.figure(figsize=(self.shape_factor*tot_width, self.shape_factor*tot_height))
        gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=width_ratios, height_ratios=height_ratios, figure=figure)
        ax = []
        for i in range(n_rows):
            ax.append([])
            for j in range(n_cols):
                # if i!=0 and j!=n_rows-1 else None
                if not (i==n_rows-1 and j==0) and ((j != n_cols-1) or (i == 0)):
                    ax[i].append(plt.subplot(gs[i, j]))
                    ax[-1][-1].set_xticks([])
                    ax[-1][-1].set_yticks([])
                else:
                    ax[i].append(None)

        # figure, ax = plt.subplots(n_rows, n_cols, figsize=(self.shape_factor*data[0].shape[-1], self.shape_factor*data[0].shape[0]))

        canvas = FigureCanvas(figure)
        figure.set_tight_layout(True)

        im = ax[0][-2].imshow(data[0], interpolation='nearest', cmap=self.cmap, aspect='auto', animated=True,
                              vmin = data.min(), vmax=data.max())

        for i, n in enumerate(xorder):
            ax[i+1][-2].title.set_text(n)
            ax[i+1][-2].imshow(self.x_sel[n][0].transpose(-1,-2), interpolation='nearest', cmap=self.sel_cmap, aspect='auto', animated=True,
                              vmin = 0, vmax=1)


        for i, n in enumerate(yorder):
            ax[0][i].title.set_text(n)
            ax[0][i].imshow(self.y_sel[n][0], interpolation='nearest', cmap=self.sel_cmap, aspect='auto', animated=True,
                              vmin = 0, vmax=1)

        if x_marks is not None:
            ax[-1][-2].set_xticks(np.arange(self.map.shape[2]))
            ax[-1][-2].set_xticklabels(x_marks, rotation=45, fontsize=self.font_size, ha="right", rotation_mode="anchor")

        if y_marks is not None:
            ax[0][0].set_yticks(np.arange(self.map.shape[1]))
            ax[0][0].set_yticklabels(y_marks, fontsize=self.font_size)

        if data.shape[0] > 1:
            ax[0][-2].title.set_text("Step: 0")

        ax[0][0].set_ylabel(self.ylabel)
        ax[-1][-2].set_xlabel(self.xlabel)

        for a in ax:
            for b in a:
                if b is not None:
                    b.set_aspect('auto', adjustable='box')

        if self.colorbar:
            # divider = make_axes_locatable(ax[0][-1])
            # cax = divider.append_axes("right", size=0.25, pad=0.1)
            cbar = plt.colorbar(im, ax[0][-1])
            # ax[0][-1].colorbar(im)
            pass

        plt.tight_layout()

        frames = []
        for i in range(data.shape[0]):
            canvas.draw()
            image_from_plot = np.array(canvas.renderer.buffer_rgba())
            frames.append(image_from_plot.reshape(figure.canvas.get_width_height()[::-1] + (4,))[:,:,:3])

            if i < data.shape[0] - 1:
                im.set_data(data[i + 1])
                ax[0][-1].title.set_text(f"Step: {i + 1}")


        video = np.stack(frames, 0)
        return np.transpose(video, (0, 3, 1, 2)), figure

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_video(name, self.to_video()[0][np.newaxis], global_step, fps = self.fps)

    def to_wandb(self):
        return wandb.Video(self.to_video()[0], fps = self.fps)


class FullMoeRelativeAttentionCore(LayerWithVisualization, LoggingLayer, RegularizedLayer, OncePerIterLayer, torch.nn.Module):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0, input_size: Optional[int] = None,
                 projection_size: Optional[int] = None, output_size: Optional[int] = None, init_std_scale: float = 1.0,
                 perplexity_reg: float = 0, share_pk: bool = True, expert_dropout: float = 0.0,
                 selection_mode: str = "sigmoid", moe_k: int = 2, q_expert: bool = True,
                 k_expert: bool = True, v_expert: bool = True, o_expert: bool = True, norm_qk_score: bool = False,
                 v_projection_size: Optional[int] = None, same_sel: bool = False,
                 qside_n_experts: Optional[int] = None, shared_experts: bool = False,
                 kq_n_experts: Optional[int] = None, separate_kq_sel: bool = False,
                 normalize_init: bool = False, normalize_retrieval: bool = False):

        super().__init__()

        self.input_size = input_size or state_size
        self.output_size = output_size or state_size
        self.pe_size = self.input_size
        self.perplexity_reg = perplexity_reg
        self.share_pk = share_pk
        self.expert_dropout = expert_dropout
        self.selection_mode = selection_mode
        self.iter = 0
        self.moe_k = moe_k
        self.norm_qk_score = norm_qk_score
        self.same_sel = same_sel
        self.shared_experts = shared_experts
        self.init_std_scale = init_std_scale
        self.normalize_init = normalize_init
        self.attention_to_visualize = []
        self.selections_to_visualize = {}

        self.is_expert = {
            "k": k_expert,
            "q": q_expert,
            "v": v_expert,
            "o": o_expert
        }
        self.n_experts = {
            "k": kq_n_experts or n_experts,
            "q": kq_n_experts or qside_n_experts or n_experts,
            "v": n_experts,
            "o": qside_n_experts or n_experts
        }

        self.separate_k_sel = separate_kq_sel or (self.n_experts["k"] != self.n_experts["v"])
        self.separate_q_sel = separate_kq_sel or (self.n_experts["q"] != self.n_experts["o"])

        self.sel_hist = {}
        self.sel_counts_100 = {}

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = projection_size or (state_size // n_heads)
        self.v_projection_size = v_projection_size or self.projection_size

        self.std_in = init_std_scale * math.sqrt(1 / self.input_size)
        std_out = init_std_scale * math.sqrt(1 / (n_heads * self.v_projection_size))

        self.create_selection_logic()

        self.src_side_maps = {"k", "v"}

        self.projections = torch.nn.ParameterDict({
            "q": self.create_param_block("q", self.input_size, self.projection_size, self.std_in),
            "k": self.create_param_block("k", self.input_size, self.projection_size, self.std_in),
            "v": self.create_param_block("v", self.input_size, self.v_projection_size, self.std_in),
            "o": self.create_param_block("o", self.v_projection_size, self.output_size, std_out),
        })

        if normalize_retrieval:
            self.norm_ret = torch.nn.LayerNorm(self.projection_size)
        else:
            self.norm_ret = lambda x: x

        self.sel_correlation = 0

        self.register_buffer("scale", torch.full([1], 1.0 / math.sqrt(self.projection_size)), persistent=False)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def get_n_copies(self, name: str):
        return self.n_heads

    def create_param_block(self, name: str, in_size: int, out_size: int, std: float):
        n_copies = self.get_n_copies(name)

        if self.is_expert[name]:
            exp_mul = 1 if self.shared_experts else n_copies
            p = torch.nn.Parameter(torch.randn(exp_mul * self.n_experts[name], in_size, out_size) * std)
            if self.normalize_init:
                self.renorm_keep_std(p, dim=0)
            return p
        else:
            if name == "o":
                in_size = n_copies * in_size
            else:
                out_size = n_copies * out_size
            return torch.nn.Parameter(torch.randn(out_size, in_size) * std)

    def create_selection_logic(self):
        sels_params = {}
        self.sel_map = {}

        def register_remap(dest: str, src: str) -> bool:
            if not (src in sels_params or src in self.sel_map):
                # src is not defined
                return False

            assert self.n_experts[src] == self.n_experts[dest]
            self.sel_map[dest] = self.sel_map.get(src, src)
            return True

        if self.is_expert["o"]:
            sels_params["o"] = self.init_sel("o", self.std_in)

        if self.is_expert["q"] and (self.separate_q_sel or not register_remap("q", "o")):
            sels_params["q"] = self.init_sel("q", self.std_in)

        if self.is_expert["v"] and ((not self.same_sel) or not register_remap("v", "o")):
            sels_params["v"] = self.init_sel("v", self.std_in)

        if self.is_expert["k"]:
            if (not (self.same_sel and self.separate_k_sel and register_remap("k", "q"))) and (self.separate_k_sel or not register_remap("k", "v")):
                sels_params["k"] = self.init_sel("k", self.std_in)

        self.selections = torch.nn.ParameterDict(sels_params)

    def init_sel(self, name: str, std: float) -> torch.nn.Parameter:
        n_copies = self.get_n_copies(name)
        n_experts = self.n_experts[name]
        sel = torch.nn.Parameter(torch.randn(n_experts*n_copies, self.input_size) * std)
        self.renorm_rows(sel)
        return sel

    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())


    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.get_n_copies("k"), -1).transpose(-2, -3)

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
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2)
        elif mask.src_length_mask is None:
            m = pm
        else:
            m = mask.src_length_mask.unsqueeze(-2).unsqueeze(-2) | pm

        return m

    def train(self, mode: bool = True):
        self.sel_hist = {}
        return super().train(mode)

    def get_lost_on_hist(self, l: List[torch.Tensor]) -> torch.Tensor:
        assert l[0].ndim == 4
        l = [t.flatten(1,2) for t in l]
        sel = torch.cat(l, -2)
        sel_d = F.log_softmax(sel, dim=-1)
        sel_d = framework.utils.distributed_ops.log_mean(sel_d, -2, sync_distributed=False)
        return self.perplexity_reg * ( - utils.entropy_l(sel_d).mean())

    def get_reg_loss(self) -> Dict[str, torch.Tensor]:
        l = super().get_reg_loss()
        for k, v in self.sel_hist.items():
            l[f"moe_att_entropy/{k}"] = self.get_lost_on_hist(v)

        self.sel_hist = {}
        return l

    def get_sel(self, t: torch.Tensor, w: torch.Tensor, name: str) -> Selection:
        n_experts = self.n_experts[name]
        n_copies = self.get_n_copies(name)

        sel = F.linear(t, w).float()
        sel = sel.view(*sel.shape[:-1], n_copies, -1)
        with torch.no_grad():
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, float('-inf'))
            else:
                sel2 = sel
            _, sel_index = sel2.topk(self.moe_k, dim=-1, sorted=False)
        sel_val = torch.gather(sel, -1, sel_index)

        if self.selection_mode == "softmax":
            sel_val = sel_val.softmax(-1)
        elif self.selection_mode == "sigmoid":
            sel_val = sel_val.sigmoid()
        else:
            raise ValueError("Unknown selection mode: " + self.selection_mode)

        exp_shift = 0 if self.shared_experts else n_experts

        sel_index_shifted = (torch.arange(n_copies, device=sel_index.device, dtype=sel_index.dtype) * exp_shift).unsqueeze(-1) + sel_index
        sel_index_pp = cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1), sel_val)

        return Selection(sel, sel_val, sel_index, sel_index_pp)

    def before_loss(self):
        self.iter += 1
        if self.iter % 100 == 0:
            for k, v in self.sel_counts_100.items():
                sorted_counts = v.sort(descending=True).values
                self.log(f"sel_counts/{k}", framework.visualize.plot.Barplot(sorted_counts, xlabel="expert", ylabel="usage count"), drop_old=True)

            self.sel_counts_100 = {}

    def exp_proj(self, x: torch.Tensor, w: torch.Tensor, sel: Selection) -> torch.Tensor:
        return cvmm(x, sel.sel_index, w)

    def compute_sel(self, curr_state: torch.Tensor, attend_to: torch.Tensor) -> Dict[str, Selection]:
        self.selection_mode
        outs = {}
        done = {}
        cross_atten = curr_state is not attend_to

        for name in (set(self.selections.keys()) | set(self.sel_map.keys())):
            name_actual = self.sel_map.get(name, name)

            # There coukd be 2 versions of everything: source side and destination side. Check if they are different,
            # and if not, use the cached version, my_id is the unique identifier for this transformation
            is_src_side = (name in self.src_side_maps) or not cross_atten
            my_id = (name_actual, is_src_side)

            cached = done.get(my_id)
            if cached is not None:
                outs[name] = cached
                continue

            # No cache, actually compute
            inp = attend_to if is_src_side else curr_state
            v = self.selections[name_actual]
            outs[name] = self.get_sel(inp, v, name)

            # Save history for regularization
            if self.perplexity_reg > 0 and self.training:
                if name not in self.sel_hist:
                    self.sel_hist[name] = []
                self.sel_hist[name].append(outs[name].raw_sel)

            # Visualize statistics
            if self.training and self.iter % 10 == 0:
                self.sel_counts_100[name] = self.sel_counts_100.get(name, 0) + \
                    F.one_hot(outs[name].raw_sel_index.flatten(), self.n_experts[name]).sum(0)

            done[my_id] = outs[name]

        return outs

    def project(self, name: str, src: torch.Tensor, sel: Dict[str, Selection]) -> torch.Tensor:
        if name in sel:
            sv = sel[name]
            if self.norm_qk_score and name in {"q", "k"}:
                sv.sel_index.reduction_weight = F.normalize(sv.sel_index.reduction_weight, p=1, dim=-1)
            return self.exp_proj(src, self.projections[name], sv)
        else:
            return F.linear(src, self.projections[name])

    def attend(self, curr_state: torch.Tensor, attend_to: torch.Tensor, pos_offset: int, v: torch.Tensor,
               k: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def attention_proj(self, att: torch.Tensor, v: torch.Tensor,
                       mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            att.masked_fill_(mask, float('-inf'))

        att = F.softmax(att, dim=-1)

        res = att @ v
        return res, att

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                pos_offset: Optional[int] = None, need_weights: bool = False):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]

        if pos_offset is None:
            assert curr_state.shape[1] == attend_to.shape[1], "If attend_to has different shape than curr_state, pos_offset should be provided"
            pos_offset = 0

        sel = self.compute_sel(curr_state, attend_to)

        # scale q and k with sqrt(scale) before the attention. This should save memory, be faster, and
        # keep the range of k and v better. It should make attention NaNs better with float16.
        scale = self.scale.sqrt()

        q = self.project("q", curr_state, sel)
        q = q * scale.type_as(q)
        k = self.project("k", attend_to, sel)
        k = k * scale.type_as(k)
        v = self.project("v", attend_to, sel)

        q = self.project_to_torch_order(q) if "q" not in sel else q.transpose(-2,-3)
        k = self.project_to_torch_order(k) if "k" not in sel else k.transpose(-2,-3)
        v = self.project_to_torch_order(v) if "v" not in sel else v.transpose(-2,-3)

        k = self.dropout(k)

        res, att = self.attend(curr_state, attend_to, pos_offset, v, k, q, self.get_mask_tensor(attend_to.shape[-2], mask))
        res = self.norm_ret(res)

        if self.visualization_enabled:
            self.attention_to_visualize.append(att[0].detach())
            for k, s in sel.items():
                if k not in self.selections_to_visualize:
                    self.selections_to_visualize[k] = []

                with torch.no_grad():
                    m = torch.zeros([*s.sel_val[0].shape[:-1], self.n_experts[k]], device=s.sel_val.device, dtype=s.sel_val.dtype)
                    m.scatter_(-1, s.raw_sel_index[0], s.sel_val[0])

                self.selections_to_visualize[k].append(m)

        if self.get_n_copies("k") != self.get_n_copies("v"):
            res = res.view(
                *res.shape[:-1], self.get_n_copies("v") // self.get_n_copies("k"), -1).transpose(2,3).flatten(1,2).contiguous()

        if self.is_expert["o"]:
            res = res.transpose(-2, -3)
            # The output selection indices are calculated from the current state and are also used for projecting "q".
            # But that projection needs to create multiple copies for the different heads. Here we already have the
            # heads, but we have to create copies for the top-k elements. We can calculate that from the reduction
            # weight. We also want to compute not only the weighted average between the top-k elements, but also
            # of the different heads. So reshape the reduction weight accordingly.
            o_sel = sel["o"].sel_index.clone()
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out = cvmm(res, o_sel, self.projections["o"])
        else:
            res = res.transpose(-2, -3)
            out = F.linear(res.contiguous().view(*curr_state.shape[:-1], -1), self.projections["o"])

        return out

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        marks = options.get("steplabel")
        n_steps = options.get("n_steps") or 9999999
        y_marks = options.get("target_labels", marks)

        ns1 = (self.attention_to_visualize[0].shape[-2] + n_steps) if n_steps < 0 else 0
        ns1_e = self.attention_to_visualize[0].shape[-2] if n_steps < 0 else n_steps
        ns2 = (self.attention_to_visualize[0].shape[-1] + n_steps) if n_steps < 0 else 0
        ns2_e = self.attention_to_visualize[0].shape[-1] if n_steps < 0 else n_steps

        if marks is not None:
            assert len(marks) == self.attention_to_visualize[0].shape[-1]
            marks = marks[ns2:ns2_e]

        if y_marks is not None:
            assert len(y_marks) == self.attention_to_visualize[0].shape[-2]
            y_marks = y_marks[ns1:ns1_e]

        if options.get("mha.plot_head_details") and self.attention_to_visualize[0].shape[0] > 1:
            for head in range(self.attention_to_visualize[0].shape[0]):
                sel_map = {k: [e[:, head][ns1:ns1_e] if k in {'q', 'o'} else e[:, head][ns2:ns2_e] for e in v] for k, v in self.selections_to_visualize.items()}
                selections = {k: torch.stack(v, 0).cpu() for k, v in sel_map.items()}

                x_selections = {k: v for k, v in selections.items() if k in {'k', 'v'}}
                y_selections = {k: v for k, v in selections.items() if k in {'q', 'o'}}

                r[f"head_{head}"] = MoEAttentionPlot(
                    torch.stack([layer[head][ns1:ns1_e, ns2:ns2_e] for _, layer in enumerate(self.attention_to_visualize)], 0),
                    x_selections,  y_selections,
                    ylabel="dest", xlabel="src", x_marks=marks, y_marks=y_marks)

        r["attention_max"] = framework.visualize.plot.AnimatedHeatmap(
            torch.stack([layer.max(0)[0][ns1:ns1_e, ns2:ns2_e] for _, layer in enumerate(self.attention_to_visualize)], 0),
            ylabel="dest", xlabel="src", textval=False, x_marks=marks, y_marks=y_marks, ignore_wrong_marks=True)

        self.attention_to_visualize = []
        self.selections_to_visualize = {}
        return r

    def dump_logs(self, save_dir: str):
        if torch.is_tensor(self.sel_correlation):
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.sel_correlation, os.path.join(save_dir, "sel_correlation.pt"))

    def get_logs(self) -> Dict[str, Any]:
        res = super().get_logs()

        if torch.is_tensor(self.sel_correlation):
            coo = self.sel_correlation / self.sel_correlation.flatten(1).sum(-1).clamp(min=1)[:, None, None]
            for h in range(self.n_heads):
                res[f"expert_coocurence_{h}"] = framework.visualize.plot.Heatmap(coo[h], xlabel="o expert", ylabel="v expert", textval=False)
            self.sel_correlation = 0
        return res


class FullMoeRelativeAttention(FullMoeRelativeAttentionCore):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0, input_size: Optional[int] = None,
                 projection_size: Optional[int] = None, output_size: Optional[int] = None, init_std_scale: float = 1.0,
                 perplexity_reg: float = 0, share_pk: bool = True, expert_dropout: float = 0.0,
                 selection_mode: str = "sigmoid", moe_k: int = 2, q_expert: bool = True,
                 k_expert: bool = True, v_expert: bool = True, o_expert: bool = True, norm_qk_score: bool = False,
                 v_projection_size: Optional[int] = None, same_sel: bool = False,
                 qside_n_experts: Optional[int] = None, shared_experts: bool = False,
                 kq_n_experts: Optional[int] = None, separate_kq_sel: bool = False,
                 normalize_init: bool = False, normalize_retrieval: bool = False):

        super().__init__(
            state_size, n_heads, n_experts, dropout, input_size, projection_size, output_size, init_std_scale,
            perplexity_reg, share_pk, expert_dropout, selection_mode, moe_k, q_expert, k_expert, v_expert,
            o_expert, norm_qk_score, v_projection_size, same_sel, qside_n_experts, shared_experts,
            kq_n_experts, separate_kq_sel, normalize_init, normalize_retrieval=normalize_retrieval)

        self.pe_size = state_size
        std_pe = init_std_scale * math.sqrt(1 / self.pe_size)
        self.pos_to_pk = torch.nn.Parameter(torch.randn(self.get_n_copies("k") * self.projection_size, self.pe_size) * std_pe)

        self.register_buffer("pos_encoding", self.create_pos_buffer(1000), persistent=False)

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

    def attend(self, curr_state: torch.Tensor, attend_to: torch.Tensor, pos_offset: int, v: torch.Tensor,
               k: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        scale = self.scale.sqrt()

        k_pos = self.get_pos_subset(attend_to.shape[-2], pos_offset) * scale
        k_pos = F.linear(k_pos, self.pos_to_pk)
        k_pos = self.project_to_torch_order(k_pos)

        k_pos = self.dropout(k_pos)

        qc = qp = q

        att = self.shift(qp @ k_pos.transpose(-2, -1)) + qc @ k.transpose(-2, -1)
        return self.attention_proj(att, v, mask)



class FullMoeRopeAttention(FullMoeRelativeAttentionCore):
    def __init__(self, state_size: int, n_heads: int, n_experts: int, dropout: float = 0.0, input_size: Optional[int] = None,
                 projection_size: Optional[int] = None, output_size: Optional[int] = None, init_std_scale: float = 1.0,
                 perplexity_reg: float = 0, share_pk: bool = True, expert_dropout: float = 0.0,
                 selection_mode: str = "sigmoid", moe_k: int = 2, q_expert: bool = True,
                 k_expert: bool = True, v_expert: bool = True, o_expert: bool = True, norm_qk_score: bool = False,
                 v_projection_size: Optional[int] = None, same_sel: bool = False,
                 qside_n_experts: Optional[int] = None, shared_experts: bool = False,
                 kq_n_experts: Optional[int] = None, separate_kq_sel: bool = False,
                 rotate_fraction: float = 0.5, rope_base: float = 10000, normalize_init: bool = False, normalize_retrieval: bool = False):

        super().__init__(
            state_size, n_heads, n_experts, dropout, input_size, projection_size, output_size, init_std_scale,
            perplexity_reg, share_pk, expert_dropout, selection_mode, moe_k, q_expert, k_expert, v_expert,
            o_expert, norm_qk_score, v_projection_size, same_sel, qside_n_experts, shared_experts,
            kq_n_experts, separate_kq_sel, normalize_init, normalize_retrieval=normalize_retrieval)

        self.n_rotate = int(rotate_fraction * self.projection_size)

        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

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

    def attend(self, curr_state: torch.Tensor, attend_to: torch.Tensor, pos_offset: int, v: torch.Tensor,
               k: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.n_rotate > 0:
            q, k = self.rotate(q, k, pos_offset or 0)

        att = q @ k.transpose(-2, -1)
        return self.attention_proj(att, v, mask)
