import lib
from common import parse_args, format_flops, format_mem, format_param_count, print_run_groups
from collections import OrderedDict
from run_tests import test_field_name



if __name__ == "__main__":
    groups = OrderedDict()
    groups["small"] = [
        "wikitext103_xl_rope",
        "wikitext103_moeatt_rope_matched_k3",
        "wikitext103_xl_rope_h2"
    ]

    groups["big"] = [
        "wikitext103_baseline_big_rope",
        "wikitext103_moeatt_big_h4_matched_k2_rope",
        "wikitext103_baseline_big_rope_h2"
    ]

    print_run_groups(groups, 75000)
