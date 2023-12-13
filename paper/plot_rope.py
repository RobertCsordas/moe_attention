import lib
from common import parse_args, format_flops

runs = [
    "wikitext103_xl_rope",
    "wikitext103_moeatt_rope_matched_k3",
]



# W&B sometimes forgets to upload summary. Also, sometimes it doesn't upload the data from the end of the training,
# including the checkpoint. But history works.
runs = list(lib.get_runs(runs, check_finished=False))
hists = [r.history(keys=["validation/val/perplexity", "iteration"], pandas=False) for r in runs]

step = min(max(s["iteration"] for s in hi) for hi in hists)
perf = [[h["validation/val/perplexity"] for h in hi if h["iteration"] == step][0] for hi in hists]


order = list(sorted(range(len(runs)), key=lambda i: perf[i]))

for o in order:
    run = runs[o]
    # Even better. Sometimes it does not upload the config! If missing, try parsing the command line.

    trafo_variant, att_variant, nh = parse_args(run, "transformer.variant", "moe.att.variant", "transformer.n_heads")
    if trafo_variant == "preln_moeatt":
        if att_variant == "moa":
            type = "MoA"
        elif att_variant in {"full", "full_rope"}:
            type = "MoE Att."
        else:
            assert False

    elif trafo_variant == "preln_relative":
        type = "Transformer XL"
    elif trafo_variant == "preln_rope":
        type = "Transofrmer"
    else:
        assert False

    print(f"{type} & {nh} & {perf[o]:.2f} & {format_flops(run)}\\\\")

print("Step:", step)
print(perf)


