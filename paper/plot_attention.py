import os
import torch


os.makedirs("att_plots/listops_moeatt", exist_ok=True)
plots = torch.load("plot_dumps_listops/listops_moeatt/iid_0000.pth")
for l in range(6):
    for h in range(2):
        plots[f"validation_plots/iid/activations/trafo.layers.{l}.self_attn/head_{h}"].to_video()[1].savefig(f"att_plots/listops_moeatt/l{l}_h{h}.pdf")
    plots[f"validation_plots/iid/activations/trafo.layers.{l}.self_attn/attention_max"].to_video()[1].savefig(f"att_plots/listops_moeatt/l{l}_max.pdf")
