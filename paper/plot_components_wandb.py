import lib
from common import parse_args, format_flops, format_mem, format_param_count


runs = [
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
]

runs = list(lib.get_runs(runs, check_finished=False))
hists = [r.history(keys=["validation/val/perplexity", "iteration"], pandas=False) for r in runs]

step = min(max(s["iteration"] for s in hi) for hi in hists)
results = [[h["validation/val/perplexity"] for h in hi if h["iteration"] == step][0] for hi in hists]

order = list(sorted(range(len(runs)), key=lambda i: results[i]))


print("\\begin{tabular}{lrccccr}")
print("\\toprule")
print("Model & $\\nheads$ & V expert & K expert & Q expert & O expert & Perplexity \\\\")
print("\\midrule")


for o in order:
    run = runs[o]
    r = ""
    if run.config["transformer.variant"] == "preln_moeatt":
        r += f"MoE-Att & " + str(run.config["transformer.n_heads"]) + " & " + " & "
        for e in ["v","k","q","o"]:
            if run.config[f"moe.att.{e}_expert"]:
                r += "Y & "
            else:
                r += "N & "
    else:
        r += f"Transformer XL & " + str(run.config["transformer.n_heads"]) + " & " + " & - & - & - & - &"

    r += f"{results[o]:.2f} \\\\"
    print(r)

print("\\bottomrule")
