import lib
from common import parse_args, format_flops, format_mem, format_param_count
from collections import OrderedDict
from run_tests import test_field_name
# from plot_datasets import print_runs
from common import print_run_groups


groups = OrderedDict()
groups["Wikitext103 small"] = [
    "wikitext103_xl",
    "wikitext103_small_fullmoe",
]

groups["Wikitext103 big"] = [
    "wikitext103_baseline_big",
    "wikitext103_big_fullmoe_h4_k2_matchfix"

]

groups["C4 small"] = [
    "c4_xl_new",
    "c4_small_fullmoe"
]

groups["C4 big"] = [
    "c4_baseline_big",
    "c4_big_fullmoe_h4_matchfix"
]

groups["PES2O small"] = [
    "pes2o_xl_new",
    "pes2o_small_fullmoe"
]

groups["PES2O big"] = [
    "pes2o_baseline_big_new",
    "pes2o_big_fullmoe_h4_matchfix"
]


print_run_groups(groups, 99000)
