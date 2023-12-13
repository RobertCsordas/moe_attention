from run_tests import get_runs_and_infos, test_field_name, dataset, n_test_blocks
import torch
import os
from collections import OrderedDict


ckpt_remap = {
     "data_to_q": "projections.q",
     "data_to_v": "projections.v",
     "data_to_k": "projections.k",
     "out_proj": "projections.o",
}

def patch_ckpt(path: str) -> str:
    out_path = path + ".patched"
    if os.path.isfile(out_path):
        return out_path

    ckpt = torch.load(path)
    print(ckpt.keys())
    patch_layers = set()
    for k, _ in ckpt["model"].items():
        if "self_attn.sel_" in k:
            patch_layers.add(".".join(k.split(".")[:-1]))


    if not patch_layers:
        return path

    new_model = OrderedDict()
    for k, v in ckpt["model"].items():
        remapped = False
        for p in patch_layers:
            if k.startswith(p):
                last_part = k[len(p) + 1:]

                remapped = True
                if last_part == "sel_dst":
                    if ckpt['run_invariants']['args']['moe.att.o_expert']:
                        new_model[p+".selections.o"] = v
                    elif ckpt['run_invariants']['args']['moe.att.q_expert']:
                        new_model[p+".selections.q"] = v
                elif last_part == "sel_src":
                    if ckpt['run_invariants']['args']['moe.att.v_expert']:
                        new_model[p+".selections.v"] = v
                    elif ckpt['run_invariants']['args']['moe.att.k_expert']:
                        new_model[p+".selections.k"] = v
                elif last_part in ckpt_remap:
                    new_model[p+"."+ckpt_remap[last_part]] = v
                else:
                    remapped = False
        if not remapped:
            new_model[k] = v

    ckpt["model"] = new_model
    ckpt["optimizer"]["state"] = {}
    ckpt["optimizer"]["param_groups"][0]["params"] = list(range(len(new_model)))

    torch.save(ckpt, out_path)

    return out_path


runs, infos = get_runs_and_infos([
    "wikitext103_moeatt_matched_voq",
    "wikitext103_moeatt_matched_vok",
    "wikitext103_moeatt_matched_qko",
    "wikitext103_moeatt_matched_qkv",
    "wikitext103_moeatt_matched_q_only",
    "wikitext103_moeatt_matched_k_only",
    "wikitext103_moeatt_matched_qk",
    "wikitext103_moeatt_matched_vq",
    "wikitext103_moeatt_matched_vk",
    "wikitext103_moeatt_matched_l_ff",
    "wikitext103_moeatt_matched_v_only",
    "wikitext103_moeatt_matched_o_only",
    "wikitext103_moeatt_matched_ko",
    "wikitext103_moeatt_matched_qo",
    "wikitext103_moeatt_matched_vkqo",
    "wikitext103_xl",
    "wikitext103_xl_h2"
], patch_ckpt=patch_ckpt, n_blocks=4, bs=64)

runs = list(runs)

results = [infos[r.id]["test_results"][test_field_name[r.config["task"]]] for r in runs]
order = list(sorted(range(len(runs)), key=lambda i: results[i]))

print("\\begin{tabular}{lcccr}")
print("\\toprule")
print("Model & $\\nheads$ & V expert & K expert & Q expert & O expert & Perplexity \\\\")
print("\\midrule")


for o in order:
    run = runs[o]
    r = ""
    if run.config["transformer.variant"] == "preln_moeatt":
        r += f"MoE-Att & " + str(run.config["transformer.n_heads"]) + " & "
        for e in ["v","k","q","o"]:
            if run.config[f"moe.att.{e}_expert"]:
                r += "Y & "
            else:
                r += "N & "
    else:
        r += f"Transformer XL & " + str(run.config["transformer.n_heads"]) + " - & - & - & - &"

    r += f"{results[o]:.2f} \\\\"
    print(r)

print("\\bottomrule")
