import lib
from common import parse_args, format_flops, format_mem, format_param_count, get_type, get_dataset, get_ordered_performance
from collections import OrderedDict
from run_tests import test_field_name


def print_hyperparams(runs):
    # W&B sometimes forgets to upload summary. Also, sometimes it doesn't upload the data from the end of the training,
    # including the checkpoint. But history works.
    perf, order = get_ordered_performance(runs, 0)

    for o in order:
        run = runs[o]
        nparams = format_param_count(run)
        # Even better. Sometimes it does not upload the config! If missing, try parsing the command line.

        type = get_type(run)
        dataset = get_dataset(run)
        nh, hps, dff, t, ne, k, nlayer, clip = parse_args(run, "transformer.n_heads", "transformer.head_projection_size", "dff", "lm.unroll", "moe.att.n_experts", "moe.att.k", "transformer.encoder_n_layers",  "grad_clip")

        if not "\\name" in type:
            ne = "-"
            k = "-"

        print(f"{type} & {dataset} & {nh} & {nparams} & {hps} & {dff} & {ne} & {k} & {t} & {nlayer} & {clip} \\\\")





if __name__ == "__main__":
    groups = OrderedDict()
    groups["C4 small"] = [
        "c4_xl",
        "c4_moeatt_small_k3",
        "c4_xl_h2",
    ]

    groups["C4 big"] = [
        "c4_baseline_big",
        "c4_moeatt_big_h4_matched",
        "c4_baseline_big_h4",
    ]

    groups["Wikitext 103 small"] = [
        "wikitext103_xl",
        "wikitext103_moeatt_matched_l_ff",
        "wikitext103_xl_h2",
    ]

    groups["Wikitext 103 big"] = [
        "wikitext103_baseline_big",
        "wikitext103_moeatt_big_h2_matched_k4",
        "wikitext103_baseline_big_h2",
    ]

    groups["PES2O small"] = [
        "pes2o_xl",
        "pes2o_moeatt_small_k3",
        "pes2o_xl_h2",
    ]

    groups["PES2O big"] = [
        "pes2o_baseline_big",
        "pes2o_moeatt_big_h4_matched_norminit",
        "pes2o_baseline_big_h4"
    ]

    groups["Enwik8"] = [
        "enwik8_baseline",
        "enwik8_baseline_h2",
        "enwik8_moeatt"
    ]

    groups["RoPE small"] = [
        "wikitext103_xl_rope",
        "wikitext103_moeatt_rope_matched_k3",
    ]

    groups["RoPE big"] = [
        "wikitext103_baseline_big_rope",
        "wikitext103_moeatt_big_h4_matched_k2_rope",
    ]

    groups["Wikitext103 small"] = [
        "wikitext103_small_fullmoe",
    ]

    groups["Wikitext103 big fullmoe"] = [
        "wikitext103_big_fullmoe_h4_k2_matchfix",
    ]

    groups["C4 small fullmoe"] = [
        "c4_small_fullmoe"
    ]

    groups["C4 big fullmoe"] = [
        "c4_big_fullmoe_h4_matchfix"
    ]

    groups["PES2O small fullmoe"] = [
        "pes2o_small_fullmoe"
    ]

    groups["PES2O big fullmoe"] = [
        "pes2o_big_fullmoe_h4_matchfix"
    ]


    all_runs = OrderedDict()
    for k, v in groups.items():
        all_runs[k] = lib.get_runs(v, check_finished=False)

    print("-" * 64)

    print("\\begin{tabular}{llrrrrr}")
    print("\\toprule")
    print("Model & Dataset & $\\nheads$ & \\#params & $\\dhead$ & $\\dff$ & E & K & T & $\\nlayers$ & $\\kappa$\\\\")
    print("\\midrule")
    for i, k in enumerate(all_runs.keys()):
        if i!=0:
            print("\\midrule")

        print_hyperparams(all_runs[k])

    print("\\bottomrule")
    print("\\end{tabular}")
    print("-" * 64)