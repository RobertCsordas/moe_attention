import lib
from common import parse_args, format_flops, format_mem, format_param_count

def print_runs(runs):
    runs = list(lib.get_runs(runs, check_finished=False))
    hists = [r.history(keys=["validation/val/perplexity", "iteration"], pandas=False) for r in runs]

    step = min(max(s["iteration"] for s in hi) for hi in hists)
    perf = [[h["validation/val/perplexity"] for h in hi if h["iteration"] == step][0] for hi in hists]


    order = list(sorted(range(len(runs)), key=lambda i: perf[i]))
    for o in order:
        run = runs[o]
        # Even better. Sometimes it does not upload the config! If missing, try parsing the command line.
        nparams = format_param_count(run)

        trafo_variant, att_variant, nh, moa_mode, selmode = parse_args(run, "transformer.variant", "moe.att.variant", "transformer.n_heads", "moa.mode", "moe.att.selection_mode")
        if trafo_variant == "preln_moeatt":
            if att_variant == "moa":
                selmode = "sigmoid" if moa_mode == "my" else "softmax"
                type = f"MoA & {selmode}"
            elif att_variant == "full":
                type = f"MoE Att. & {selmode}"
            else:
                assert False

        elif trafo_variant == "preln_relative":
            type = "Transformer XL & -"
        elif trafo_variant == "preln_frel_group":
            type = "GQA & -"
        else:
            assert False

        print(f"{type} & {nh} & {nparams} & {perf[o]:.2f} & {format_flops(run)} & {format_mem(run)}\\\\")


# W&B sometimes forgets to upload summary. Also, sometimes it doesn't upload the data from the end of the training,
# including the checkpoint. But history works.

print("\\begin{tabular}{lllrrrr}")
print("\\toprule")
print("Model & sel. mode & $\\nheads$ $ \\#params & Perplexity & MACs & Mem (floats) \\\\")
print_runs([
    "wikitext103_xl",
    "wikitext103_moa_matched_h_official",
    "wikitext103_moa_matched_h",
    "wikitext103_moa_matched_h2",
    "wikitext103_moeatt_matched_l_ff",
    "wikitext103_xl_groupatt_h2"
])
print("\\midrule")
print_runs([
    "wikitext103_big_moa_heads_official",
    "wikitext103_big_moa_h2_official",
    "wikitext103_big_moa_heads_official_h8",
    "wikitext103_big_moa_heads",
    "wikitext103_big_moa_h2",
    "wikitext103_moeatt_big_h2_matched_k4",
    "wikitext103_baseline_big",
])
print("\\bottomrule")

print("Step:", step)
print(perf)


