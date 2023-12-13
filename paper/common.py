import wandb
from wandb_gql import gql
from run_tests import test_field_name
from collections import OrderedDict
import lib
import os
import torch


defaults = {
    "lm.trafo.context_blocks": 1,
    "moe.att.k": 2,
    "moe.att.q_expert": 1,
    "moe.att.k_expert": 1,
    "moe.att.o_expert": 1,
    "moe.att.v_expert": 1,
    "moa.mode": "my",
    "moe.att.selection_mode": "sigmoid",
    "moe.att.variant": "moa",
    "moe.att.n_experts": 4,
    "moe.att.enable": False,
    "transformer.ff_multiplier": 2,
    "grad_clip": 1,
    "transformer.n_attention_groups": 1
}

def get_hps(run):
    state_size, n_heads = parse_args(run, "state_size", "transformer.n_heads")
    return int(state_size) // int(n_heads)

def get_dff(run):
    state_size, ffm = parse_args(run, "state_size", "transformer.ff_multiplier")
    return int(int(state_size) * float(ffm))


special_args = {
    "transformer.head_projection_size": get_hps,
    "dff": get_dff,
}

def parse_args(run, *args):
    res = []
    for a in args:
        if a in run.config:
            found = run.config[a]
            # res.append(run.config[a])
        else:
            for m in run.metadata["args"]:
                if not m.startswith("--"):
                    continue

                m = m[2:].split("=")
                assert len(m) == 2
                if m[0] == a:
                    found = m[1]
                    # res.append(m[1])
                    break
            else:
                if a in defaults:
                    found = defaults[a]
                elif a in special_args:
                    found = special_args[a](run)
                    # res.append(special_args[a](run))
                else:
                    assert False, f"Arg {a} not found"

        if isinstance(found, str) and found.lower()=="none" and a in special_args:
            found = special_args[a](run)

        res.append(found)
    return res

def get_attention_flops(run):
    trafo_var, att_var, state_size, n_heads, unroll, hps, cblocks, att_k, q_e, k_e, v_e, o_e, n_experts, att_en, n_att_groups = parse_args(run, "transformer.variant", "moe.att.variant", "state_size", "transformer.n_heads", "lm.unroll", "transformer.head_projection_size", "lm.trafo.context_blocks", "moe.att.k", "moe.att.q_expert", "moe.att.k_expert", "moe.att.v_expert", "moe.att.o_expert", "moe.att.n_experts", "moe.att.enable", "transformer.n_attention_groups")

    if trafo_var == "preln_moe":
        if att_en:
            trafo_var = "preln_moeatt"
        else:
            trafo_var = "preln_relative"

    state_size = int(state_size)
    n_heads = int(n_heads)
    unroll = int(unroll)
    hps = int(hps) if hps!="none" else (state_size // n_heads)
    cblocks = int(cblocks)
    att_k = int(att_k)
    n_experts = int(n_experts)

    if trafo_var == "preln_relative":
        return n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + 4 * state_size * n_heads * hps * unroll + 2 * n_heads * state_size * unroll * hps * (1+cblocks)
    elif trafo_var == "preln_frel_group":
        return n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + \
              2 * state_size * (n_heads + n_att_groups) * hps * unroll + \
              2 * n_att_groups * state_size * unroll * hps * (1+cblocks)
    elif trafo_var == "preln_moeatt":
        if att_var in {"full", "full_rope"}:
            n_exp = int(q_e) + int(k_e) + int(v_e) + int(o_e)
            n_total_proj = n_exp * att_k + 4 - n_exp

            n_sels = min(1, int(k_e) + int(v_e)) + min(1, int(q_e) + int(o_e))

            res = n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + n_total_proj * state_size * n_heads * hps * unroll + n_exp*n_heads*att_k*unroll*hps + n_sels * n_experts * state_size * unroll * (1+cblocks)
            if att_var == "full":
                res += 2 * n_heads * state_size * unroll * hps
            return res
        elif att_var == "moa":
            return n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + (2*n_heads + 2) * state_size * hps * unroll  + n_heads*att_k*unroll*hps + 1 * n_experts * state_size * unroll + 2 * state_size * unroll * hps * (1+cblocks)
        else:
            assert False
    elif trafo_var == "preln_rope":
        return n_heads * hps * unroll ** 2 * (1+cblocks) * 2 + 4 * state_size * n_heads * hps * unroll
    else:
        assert False
    # print(state_size, n_heads, unroll, hps)



def get_attention_mem(run):
    trafo_var, att_var, state_size, n_heads, unroll, hps, cblocks, att_k, q_e, k_e, v_e, o_e, n_experts, att_en, n_att_groups = parse_args(run, "transformer.variant", "moe.att.variant", "state_size", "transformer.n_heads", "lm.unroll", "transformer.head_projection_size", "lm.trafo.context_blocks", "moe.att.k", "moe.att.q_expert", "moe.att.k_expert", "moe.att.v_expert", "moe.att.o_expert", "moe.att.n_experts", "moe.att.enable", "transformer.n_attention_groups")

    if trafo_var == "preln_moe":
        if att_en:
            trafo_var = "preln_moeatt"
        else:
            trafo_var = "preln_relative"

    state_size = int(state_size)
    n_heads = int(n_heads)
    unroll = int(unroll)
    hps = int(hps) if hps!="none" else (state_size // n_heads)
    cblocks = int(cblocks)
    att_k = int(att_k)
    n_experts = int(n_experts)

    if trafo_var in "preln_relative":
        return n_heads * unroll ** 2 * (1+cblocks) * 2 + 4 * n_heads * hps * unroll + 2 * n_heads * unroll * hps * (1+cblocks)
    elif trafo_var == "preln_frel_group":
        return n_heads * unroll ** 2 * (1+cblocks) * 2 + \
               2 * (n_heads + n_att_groups) * hps * unroll + \
               2 * n_att_groups * unroll * hps * (1+cblocks)
    elif trafo_var == "preln_moeatt":
        if att_var in {"full", "full_rope"}:
            # K doesn't matter for the memory usage (with a smart kernel)
            n_sels = min(1, int(k_e) + int(v_e)) + min(1, int(q_e) + int(o_e))

            res = n_heads * unroll ** 2 * (1+cblocks) * 2 + 4 * n_heads * hps * unroll + n_sels * n_experts * unroll * (1+cblocks)
            if att_var == "full":
                res += 2 * n_heads * unroll * hps
            return res
        elif att_var == "moa":
            return n_heads * unroll ** 2 * (1+cblocks) * 2 + (2*n_heads + 2) * hps * unroll + 2 * unroll * hps + 1 * n_experts * unroll * (1+cblocks)
        else:
            assert False
    elif trafo_var == "preln_rope":
        return n_heads * unroll ** 2 * (1+cblocks) * 2 + 4 * n_heads * hps * unroll
    else:
        assert False
    # print(state_size, n_heads, unroll, hps)


def format_flops(run):
    flp = get_attention_flops(run)/1e6

    if flp > 1000:
        return f"{flp/1000:.1f}G"
    else:
        return f"{flp:.1f}M"


def format_mem(run):
    flp = get_attention_mem(run)/1e6

    if flp > 1000:
        return f"{flp/1000:.1f}G"
    else:
        return f"{flp:.1f}M"


def get_logs(run):
    QUERY = gql('''
    query RunLogLines($projectName: String!, $entityName: String, $runName: String!) {
        project(name: $projectName, entityName: $entityName) {
            id
            run(name: $runName) {
                id
                logLines(last: 1000000) {
                    edges {
                        node {
                            line
                            }
                    }
                }
            }
        }
    }
    ''')


    response = run.client.execute(QUERY, variable_values={
        'entityName': run.entity,
        'projectName': run.project,
        'runName': run.id,
    })

    logs = []
    for l in response["project"]["run"]["logLines"]["edges"]:
        line = l["node"]["line"]
        logs.append(line.strip())
    return logs


def get_param_count(run):
    if "n_params" in run.summary:
        return run.summary["n_params"]

    for l in get_logs(run):
        if "Total number of model parameters" in l:
            return int(l.split(":")[-1].strip())
    assert False


def format_param_count(run):
    n = get_param_count(run) / 1e6
    return f"{int(round(n))}M"


def get_type(run):
    trafo_variant, att_variant = parse_args(run, "transformer.variant", "moe.att.variant")

    if trafo_variant == "preln_moeatt":
        if att_variant == "moa":
            type = "MoA"
        elif att_variant == "full":
            type = "{\\nameshort}"
        elif att_variant == "full_rope":
            type = "{\\nameshort} (RoPE)"
        else:
            assert False
    elif trafo_variant == "preln_moe":
        type = "{\\namefullmoeshort}"
    elif trafo_variant == "preln_relative":
        type = "Transformer XL"
    elif trafo_variant == "preln_frel_group":
        type = "GQA"
    elif trafo_variant == "preln_rope":
        type = "Transformer (RoPE)"
    else:
        assert False

    return type


def get_dataset(run):
    task, = parse_args(run, "task")
    if "wikitext103" in task:
        dataset = "Wikitext 103"
    elif "c4" in task:
        dataset = "C4"
    elif "pes2o" in task:
        dataset = "peS2o"
    elif "enwik8" in task:
        dataset = "Enwik8"
    else:
        assert False

    return dataset


def patch_ckpt(path: str) -> str:
    ckpt_remap = {
        "data_to_q": "projections.q",
        "data_to_v": "projections.v",
        "data_to_k": "projections.k",
        "out_proj": "projections.o",
    }

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


def get_ordered_performance(runs, minstep=95000):
    task, = parse_args(runs[0], "task")
    tfield = "validation/"+test_field_name[task]
    hists = [r.history(keys=[tfield, "iteration"], pandas=False) for r in runs]

    step = min(max(s["iteration"] for s in hi) for hi in hists)
    assert step > minstep
    perf = [[h[tfield] for h in hi if h["iteration"] == step][0] for hi in hists]

    order = list(sorted(range(len(runs)), key=lambda i: perf[i]))

    return perf, order


def print_runs(runs, minstep=99000):
    # W&B sometimes forgets to upload summary. Also, sometimes it doesn't upload the data from the end of the training,
    # including the checkpoint. But history works.
    perf, order = get_ordered_performance(runs, minstep)

    for o in order:
        run = runs[o]
        nparams = format_param_count(run)
        # Even better. Sometimes it does not upload the config! If missing, try parsing the command line.

        type = get_type(run)
        dataset = get_dataset(run)
        nh, = parse_args(run, "transformer.n_heads")

        print(f"{type} & {dataset} & {nh} & {nparams} & {perf[o]:.2f} & {format_flops(run)} & {format_mem(run)}\\\\")


def print_run_groups(groups, minstep=95000, test: bool = False):
    all_runs = OrderedDict()
    for k, v in groups.items():
        all_runs[k] = lib.get_runs(v, check_finished=False)

    print("-" * 64)
    print("\\begin{tabular}{llrrrrrrr}")
    print("\\toprule")
    print("Model & Dataset & $\\nheads$ & \\#params & ppl/bpc & MACs & Mem (floats) \\\\")
    print("\\midrule")

    for i, k in enumerate(all_runs.keys()):
        if i!=0:
            print("\\midrule")

        print_runs(all_runs[k], minstep=minstep)

    print("\\bottomrule")
    print("\\end{tabular}")


    print("-" * 64)
