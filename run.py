#!/usr/bin/python3

import yaml
import sys
import subprocess
import os

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <the yaml file to run> <optinal additional arguments>")
    sys.exit(-1)

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

args = []

for p, pval in config["parameters"].items():
    if p in ["log", "sweep_id_for_grid_search"]:
        continue

    args.append("-" + p)
    if "value" in pval:
        assert "values" not in pval
        args.append(pval["value"])
    elif "values" in pval:
        if len(pval["values"]) == 1:
            args.append(pval["values"][0])
        else:
            while True:
                print(f"Choose value for \"{p}\"")
                for i, v in enumerate(pval["values"]):
                    print(f"  {i+1}: {v}")

                choice = input("> ")
                if not choice.isdigit() or int(choice) < 1 or int(choice) > len(pval["values"]):
                    print("Invalid choice.")
                    continue

                args.append(pval["values"][int(choice) - 1])
                break

if "name" not in config["parameters"]:
    args.append("-name")
    args.append(os.path.basename(sys.argv[1]).replace(".yaml", ""))

replace = {
    "${env}": "",
    "${program}": config["program"],
    "${args}": " ".join([str(a) for a in args])
}

cmd = (" ".join([replace.get(c, c) for c in config["command"]])).strip() + " " + " ".join(sys.argv[2:])
print(f"Running {cmd}")
subprocess.run(cmd, shell=True)
