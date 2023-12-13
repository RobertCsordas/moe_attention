from common import print_run_groups
from collections import OrderedDict



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

    print_run_groups(groups)
