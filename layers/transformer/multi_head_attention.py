import torch
import torch.nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, List, Union, Tuple, Dict, Any
from dataclasses import dataclass
from ..layer_with_visualization import LayerWithVisualization
import framework


@dataclass
class AttentionMask:
    src_length_mask: Optional[torch.Tensor]
    position_mask: Optional[torch.Tensor]


class MultiHeadAttentionBase(LayerWithVisualization):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.1, projection_size: Optional[int] = None):
        if projection_size is None:
            assert state_size % n_heads == 0
        super().__init__()
        self.attention_to_visualize = []

        self.state_size = state_size
        self.projection_size = projection_size or (state_size // n_heads)
        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(self.projection_size)

        self.dropout = torch.nn.Dropout(dropout)

    @staticmethod
    def apply_logit_masks(logits: torch.Tensor, mask: Optional[AttentionMask], val: float = float("-inf")) -> torch.Tensor:
        if mask is None:
            return logits

        if mask.position_mask is not None:
            # [..., N_out, N_in], broadcast works
            if mask.position_mask.shape[-1] < logits.shape[-1]:
                # if mask is only applied to the end. For example previous context from language models
                logits = torch.cat([
                    logits[..., :-mask.position_mask.shape[-1]],
                    logits[..., -mask.position_mask.shape[-1]:].masked_fill(mask.position_mask, val)
                ], -1)
            else:
                logits = logits.masked_fill(mask.position_mask, val)

        if mask.src_length_mask is not None:
            # [B, ...., N_in], needs manual shaping
            b, i = mask.src_length_mask.shape
            pad_dims = logits.ndim - 2
            logits = logits.masked_fill(mask.src_length_mask.view([b] + [1] * pad_dims + [i]), val)

        return logits

    def _masked_softmax(self, logits: torch.Tensor, mask: Optional[AttentionMask]) -> torch.Tensor:
        if mask is None or (mask.src_length_mask is None and mask.position_mask is None):
            return F.softmax(logits, -1)

        # Output shape: [n_batch * n_heads, n_time_dest, n_time_src]
        bb, n_time_dest, n_time_src = logits.shape

        logits = logits.view(bb // self.n_heads, self.n_heads, n_time_dest, n_time_src)
        logits = self.apply_logit_masks(logits, mask)

        logits = F.softmax(logits, -1)
        return logits.view(bb, n_time_dest, n_time_src)

    def _attention_read(self, mask: Optional[AttentionMask], scores: torch.Tensor, v: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        # scores: [n_batch * n_heads, n_out, n_in]
        # v: [n_nbatch * n_heads, n_in]
        # Output data shape [n_batch * n_heads, n_time_dest, data_size]
        # Out attention score shape: [n_batch, n_heads, n_time_dest, n_time_src]
        s_reshape = scores.view(-1, self.n_heads, *scores.shape[1:])
        # scores = self.dropout(scores)
        if self.visualization_enabled:
            self.attention_to_visualize.append(s_reshape[0].detach())
        return torch.bmm(scores, v), s_reshape

    def transform_data(self, input: torch.Tensor, proj: Callable[[torch.Tensor], torch.Tensor],
                       n_projs: int) -> List[torch.Tensor]:
        # Input shape: [n_batch, n_steps, n_channels]
        # Output: Tuple of n_projs tensors of dimension: [n_batch * n_heads, n_steps, projection_size]
        n_batch, n_steps, _ = input.shape
        transformed = proj(input).view(n_batch, n_steps, self.n_heads, n_projs, -1). \
            permute(0, 2, 1, 3, 4).contiguous().view(n_batch * self.n_heads, n_steps, n_projs, -1)
        return transformed.unbind(dim=2)

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        marks = options.get("steplabel")
        y_marks = options.get("target_labels", marks)
        n_steps = options.get("n_steps") or 9999999

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
                r[f"head_{head}"] = framework.visualize.plot.AnimatedHeatmap(
                    torch.stack([layer[head][ns1:ns1_e, ns2:ns2_e] for _, layer in enumerate(self.attention_to_visualize)], 0),
                    ylabel="dest", xlabel="src", textval=False, x_marks=marks, y_marks=y_marks, ignore_wrong_marks=True)

        r["attention_max"] = framework.visualize.plot.AnimatedHeatmap(
            torch.stack([layer.max(0)[0][ns1:ns1_e, ns2:ns2_e] for _, layer in enumerate(self.attention_to_visualize)], 0),
            ylabel="dest", xlabel="src", textval=False, x_marks=marks, y_marks=y_marks, ignore_wrong_marks=True)
        self.attention_to_visualize = []
        return r


class AttentionMergeMixin:
    state_size: int
    n_heads: int
    projection_size: int
    _attention: Callable

    def __init__(self, out_size: Optional[int]) -> None:
        self.output_size = out_size or self.state_size
        self.multi_head_merge = torch.nn.Linear(self.n_heads * self.projection_size, self.output_size, bias=False)

    def merged_attention(self, n_batch: int, n_out_steps: int, *args, need_weights: bool = False, **kwargs
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        data, scores = self._attention(*args, **kwargs)

        data = data.view(n_batch, self.n_heads, n_out_steps, -1).permute(
            0, 2, 1, 3).contiguous().view(n_batch, n_out_steps, -1)

        return self.multi_head_merge(data), scores

    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.multi_head_merge.weight)


class AbsPosAttentionBase(MultiHeadAttentionBase):
    def get_attention_scores(self, mask: Optional[AttentionMask], q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        logits = torch.bmm(q, k.transpose(1, 2))
        return self._masked_softmax(logits * self.scale, mask)

    def _attention(self, mask: Optional[AttentionMask], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        # all inputs should have a shape of [n_batch, n_steps, data_size]
        # Output shape [n_batch * n_heads, n_time_dest, data_size]
        scores = self.get_attention_scores(mask, q, k)
        return self._attention_read(mask, scores, v)


class MultiHeadAttention(AttentionMergeMixin, AbsPosAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.1, input_size: Optional[int] = None,
                 out_size: Optional[int] = None):
        super(AbsPosAttentionBase, self).__init__(state_size, n_heads, dropout)

        self.data_to_kv = torch.nn.Linear(state_size, 2 * n_heads * self.projection_size, bias=False)
        self.data_to_q = torch.nn.Linear(input_size or state_size, n_heads * self.projection_size, bias=False)

        super(MultiHeadAttention, self).__init__(out_size)
        self.reset_parameters()

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                need_weights: bool = False):
        # Input and output shape: [n_batch, n_steps, data_size]
        k, v = self.transform_data(attend_to, self.data_to_kv, 2)
        q, = self.transform_data(curr_state, self.data_to_q, 1)

        data, scores = self.merged_attention(curr_state.shape[0], q.shape[1], mask, q, k, v)
        if need_weights:
            return data, scores
        else:
            return data

    def reset_parameters(self):
        # super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight)
