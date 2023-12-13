import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .multi_head_attention import AttentionMask, MultiHeadAttentionBase, AttentionMergeMixin
import framework
import math
from matplotlib import cm


def shift(posmat: torch.Tensor) -> torch.Tensor:
    # Slice out a matrix diagonally. Each successive row is sliced one position to the left compared.
    # shape: [n_batch, n_head, n_out, n_in * 2 - 1]
    # return: [n_batch, n_head, n_out, n_in]
    p = F.pad(posmat, (0, 1, 0, 1)).flatten(-2)  # [n_batch, n_head, (n_out + 1) * n_in * 2]
    p = p.narrow(-1, posmat.shape[-1] // 2, posmat.shape[-1] * posmat.shape[-2]).view_as(posmat)

    return p.narrow(-1, 0, (posmat.shape[-1] + 1) // 2)


class RelativeAttentionBase(MultiHeadAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float, projection_size: Optional[int] = None):
        super().__init__(state_size, n_heads, dropout=dropout, projection_size=projection_size)
        self.scale = torch.nn.Parameter(torch.tensor([self.scale]))
        self.s_bias = torch.nn.Parameter(torch.tensor([0.0]))
        self.vis_pos_vs_content = []

    def get_attention_scores(self, mask: Optional[torch.Tensor],
                   q_content: torch.Tensor, k_content: torch.Tensor,
                   q_pos: torch.Tensor, k_pos: torch.Tensor,
                   pos_offset: int, ar_gate: Optional[torch.Tensor] = None) -> torch.Tensor:

        # shape of q_content, q_pos, k_pos: [n_batch * n_heads, n_steps, data_size]
        # k_pos: [n_heads, n_in * 2 - 1, data_size]
        # ar_gate: [n_batch*n_heads, n_out, 1]
        # Output shape [n_batch * n_heads, n_out, data_size]

        n_batch = q_content.shape[0] // self.n_heads
        n_out_steps = q_content.shape[1]

        # content-content addressing
        content = torch.bmm(q_content, self.dropout(k_content).transpose(1, 2))

        # content-pos addressing.
        pos = torch.matmul(q_pos.view(n_batch, self.n_heads, n_out_steps, -1), self.dropout(k_pos).transpose(-1, -2))  # [n_batch, n_head, n_out, n_in * 2 - 1]
        fpos = shift(pos).flatten(0, 1)
        if ar_gate is not None:
            fpos = fpos * ar_gate + pos.flatten(0, 1)[..., fpos.shape[-1] - 1:] * (1 - ar_gate)

        # return self._masked_softmax((fpos) * self.scale, mask)
        if self.visualization_enabled:
            self.vis_pos_vs_content.append((content.view(n_batch, self.n_heads, *content.shape[1:])[0] * self.scale,
                                            fpos.view(n_batch, self.n_heads, *fpos.shape[1:])[0] * self.scale))

        att = (content + fpos) * self.scale
        del content, fpos, pos
        return self._masked_softmax(att, mask)

    def _attention(self, mask: Optional[torch.Tensor],
                   q_content: torch.Tensor, k_content: torch.Tensor,
                   q_pos: torch.Tensor, k_pos: torch.Tensor,
                   v: torch.Tensor, pos_offset: int,
                   ar_gate: Optional[torch.Tensor] = None) -> [torch.Tensor, torch.Tensor]:

        scores = self.get_attention_scores(mask, q_content, k_content, q_pos, k_pos, pos_offset, ar_gate)

        # Scores shape: [n_batch * n_heads, n_out, n_in]
        return self._attention_read(mask, scores, v)

    def _get_pos_subset(self, pos_encoding: torch.Tensor, length: int, offset: int) -> torch.Tensor:
        l_slice = 2 * length - 1
        assert pos_encoding.shape[0] > l_slice
        return pos_encoding.narrow(0, pos_encoding.shape[0] // 2 - length + 1 - offset, 2 * length - 1)

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        marks = options.get("steplabel")
        if options.get("mha.plot_head_details") and self.vis_pos_vs_content:
            for head in range(self.vis_pos_vs_content[0][0].shape[0]):
                cont = torch.stack([layer[0][head] for _, layer in enumerate(self.vis_pos_vs_content)], 0)
                pos = torch.stack([layer[1][head] for _, layer in enumerate(self.vis_pos_vs_content)], 0)
                i = torch.stack([layer[head] for _, layer in enumerate(self.attention_to_visualize)], 0)
                content = torch.stack([cont, pos], -1).softmax(-1)[..., 0]
                # color = torch.cat([color, torch.zeros_like(c1).unsqueeze(-1)], -1)

                color = cm.get_cmap("brg")(content.cpu().numpy())
                color[..., -1] = (i * 0.95 + 0.05).cpu().numpy()
                # color = color * i + (1 - i)

                r[f"content_vs_pos_{head}"] = framework.visualize.plot.AnimatedHeatmap(color, ylabel="dest",
                    xlabel="src", textval=False, x_marks=marks, y_marks=marks, cmap="brg", colorbar=True,
                    colorbar_ticks=[0, 0.99], colorbar_labels=["pos", "con"], ignore_wrong_marks=True)

        # r["attention_max"] = framework.visualize.plot.AnimatedHeatmap(
        #     torch.stack([layer.max(0)[0] for _, layer in enumerate(self.attention_to_visualize)], 0),
        #     ylabel="dest", xlabel="src", textval=False, x_marks=marks, y_marks=marks)
        self.vis_pos_vs_content = []

        r.update(super().plot(options))
        return r



class FixedRelativeMultiheadAttentionBase(RelativeAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, input_size: Optional[int] = None,
                 projection_size: Optional[int] = None, pos_clamp: Optional[int] = None,
                 test_pos_clamp: Optional[int] = None):
        super().__init__(state_size, n_heads, dropout, projection_size)

        self.input_size = state_size if input_size is None else input_size

        self.pos_clamp = pos_clamp
        self.test_pos_clamp = test_pos_clamp or pos_clamp

        self.pos_to_pq = torch.nn.Linear(state_size, self.n_heads * self.projection_size, bias=False)
        self.register_buffer("pos_encoding", self._create_buffer(1000, self.pos_clamp), persistent=False)
        self.register_buffer("pos_encoding_test", self._create_buffer(1000, self.test_pos_clamp), persistent=False)

    def _create_buffer(self, max_len: int, clamp: Optional[int] = None):
        l = max_len if clamp is None else min(max_len, clamp)
        res = framework.layers.sinusoidal_pos_embedding(self.state_size, 2 * l - 1, -l + 1,
                                                         device=self.pos_to_pq.weight.device)

        if clamp is not None:
            r = max_len - clamp
            if r > 0:
                res = F.pad(res.unsqueeze(0), (0, 0, r, r), mode='replicate').squeeze()

        assert res.shape[0] == 2 * max_len - 1
        return res


    def get_pos(self, l: int, offset: int) -> torch.Tensor:
        if self.pos_encoding.shape[0] < 2 * (l + offset) - 1:
            self.pos_encoding = self._create_buffer(int(2**math.ceil(math.log2(2 * (l + offset) - 1))), self.pos_clamp)
            self.pos_encoding_test = self._create_buffer(int(2**math.ceil(math.log2(2 * (l + offset) - 1))), self.test_pos_clamp)

        pos_enc = self.pos_encoding if self.training else self.pos_encoding_test
        return self.pos_to_pq(self._get_pos_subset(pos_enc, l, offset))


class FixedRelativeMultiheadAttention(AttentionMergeMixin, FixedRelativeMultiheadAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0, global_pos_bias: bool = True,
                 global_content_bias: bool = True, input_size: Optional[int] = None, absolute_gate: bool = False,
                 projection_size: Optional[int] = None, output_size: Optional[int] = None, pos_clamp: Optional[int] = None,
                 test_pos_clamp: Optional[int] = None):
        super(AttentionMergeMixin, self).__init__(state_size, n_heads, dropout, input_size,
                projection_size=projection_size, pos_clamp=pos_clamp, test_pos_clamp=test_pos_clamp)

        self.data_to_kv = torch.nn.Linear(state_size, 2 * n_heads * self.projection_size, bias=False)
        self.data_to_q = torch.nn.Linear(self.input_size, n_heads * self.projection_size, bias=False)
        self.data_to_absgate = torch.nn.Linear(self.input_size, n_heads) if absolute_gate else None

        self.global_content_bias = torch.nn.Parameter(torch.zeros([n_heads, self.projection_size])) \
                                   if global_content_bias else None
        self.global_pos_bias = torch.nn.Parameter(torch.zeros([n_heads, self.projection_size])) \
                               if global_pos_bias else None

        super(FixedRelativeMultiheadAttention, self).__init__(output_size)
        self.reset_parameters()

    def add_head_specific_bias(self, data: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        # data [batch * n_heads, len, c]
        # bias [n_heads, c]
        return (data.view(-1, bias.shape[0], *data.shape[1:]) + bias.unsqueeze(1).type_as(data)).view_as(data) \
               if bias is not None else data

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                pos_offset: Optional[int] = None, need_weights: bool = False):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]
        batch_size, in_len = attend_to.shape[0:2]
        out_len = curr_state.shape[1]

        if pos_offset is None:
            assert curr_state.shape[1] == attend_to.shape[1], "If attend_to has different shape than curr_state, pos_offset should be provided"
            pos_offset = 0

        k_content, v = self.transform_data(attend_to, self.data_to_kv, 2)
        q, = self.transform_data(curr_state, self.data_to_q, 1)

        k_pos = self.get_pos(in_len, pos_offset).view(-1, self.n_heads, self.projection_size).\
                transpose(0, 1)  # n_heads, 2*in_len -1 , projection_size

        q_content = self.add_head_specific_bias(q, self.global_content_bias)
        q_pos = self.add_head_specific_bias(q, self.global_pos_bias)


        absgate = torch.sigmoid(self.transform_data(curr_state, self.data_to_absgate, 1)[0]) \
                  if self.data_to_absgate is not None else None

        data, scores = self.merged_attention(batch_size, out_len, mask, q_content, k_content, q_pos, k_pos, v,
                                             pos_offset, ar_gate=absgate, need_weights=need_weights)

        if need_weights:
            return data, scores
        else:
            return data

    def reset_parameters(self):
    #     # super().reset_parameters()

        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.pos_to_pq.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight)

        if self.global_content_bias is not None:
            self.global_content_bias.data.fill_(0)

        if self.global_pos_bias is not None:
            self.global_pos_bias.data.fill_(0)
