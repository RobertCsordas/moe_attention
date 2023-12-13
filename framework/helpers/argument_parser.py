import os
import json
import argparse
import re
from ..data_structures.dotdict import create_recursive_dot_dict


def none_parser(other_parser):
    def fn(x):
        if x.lower() == "none":
            return None

        return other_parser(x)

    return fn


class ArgumentParser:
    _type = type

    @staticmethod
    @none_parser
    def int_list_parser(x):
        return [int(a) for a in re.split("[,_ ;]", x) if a]

    @staticmethod
    @none_parser
    def str_list_parser(x):
        return [x for x in x.split(",") if x]

    @staticmethod
    @none_parser
    def int_or_none_parser(x):
        return int(x)

    @staticmethod
    @none_parser
    def float_or_none_parser(x):
        return float(x)

    @staticmethod
    @none_parser
    def float_list_parser(x):
        return [float(a) for a in re.split("[,_ ;]", x) if a]

    @staticmethod
    @none_parser
    def float_params_parser(x):
        chunks = re.split("[, ;]", x)
        res = {}
        for c in chunks:
            if not c:
                continue

            a = c.split("=")
            if len(a) != 2:
                raise ValueError(f"Invalid format for float params: {c}")

            res[a[0]] = float(a[1])
        return res

    @staticmethod
    def _merge_args(args, new_args, arg_schemas):
        for name, val in new_args.items():
            old = args.get(name)
            if old is None:
                args[name] = val
            else:
                args[name] = arg_schemas[name]["updater"](old, val)

    class Profile:
        def __init__(self, name, args=None, include=[]):
            assert not (args is None and not include), "One of args or include must be defined"
            self.name = name
            self.args = args
            if not isinstance(include, list):
                include = [include]
            self.include = include

        def get_args(self, arg_schemas, profile_by_name):
            res = {}

            for n in self.include:
                p = profile_by_name.get(n)
                assert p is not None, "Included profile %s doesn't exists" % n

                ArgumentParser._merge_args(res, p.get_args(arg_schemas, profile_by_name), arg_schemas)

            ArgumentParser._merge_args(res, self.args, arg_schemas)
            return res

    def __init__(self, description=None, get_train_dir=lambda x: os.path.join("save", x.name)):
        self.parser = argparse.ArgumentParser(description=description)
        self.profiles = {}
        self.args = {}
        self.raw = None
        self.parsed = None
        self.get_train_dir = get_train_dir
        self.parser.add_argument("-profile", "--profile", type=str, help="Pre-defined profiles.")

    def add_argument(self, name, type=None, default=None, help="", save=True, parser=lambda x: x,
                     updater=lambda old, new: new, choice=[]):
        assert name not in ["profile"], "Argument name %s is reserved" % name
        assert not (type is None and default is None), "Either type or default must be given"

        if type is None:
            type = ArgumentParser._type(default)

        if name[0] == '-':
            name = name[1:]

        a = {
            "type": type,
            "default": int(default) if type == bool else default,
            "save": save,
            "parser": parser,
            "updater": updater,
            "choice": choice
        }

        if name in self.args:
            for k, v in self.args[name].items():
                assert a[k] == v, f"Trying to re-register argument {name} with different definition"
            return

        self.args[name] = a
        self.parser.add_argument("-" + name, "--" + name, type=int if type == bool else type, default=None, help=help)

    def add_profile(self, prof):
        if isinstance(prof, list):
            for p in prof:
                self.add_profile(p)
        else:
            self.profiles[prof.name] = prof

    def do_parse_args(self, loaded={}):
        self.raw = self.parser.parse_args()

        profile = {}
        if self.raw.profile:
            if loaded:
                if self.raw.profile != loaded.get("profile"):
                    assert False, "Loading arguments from file, but a different profile is given."
            else:
                for pr in self.raw.profile.split(","):
                    p = self.profiles.get(pr)
                    assert p is not None, "Invalid profile: %s. Valid profiles: %s" % (pr, self.profiles.keys())
                    p = p.get_args(self.args, self.profiles)
                    self._merge_args(profile, p, self.args)

        for k, v in self.raw.__dict__.items():
            if v is None:
                if k in ["profile"]:
                    self.raw.__dict__[k] = loaded.get(k)
                    continue

                if k in loaded and self.args[k]["save"]:
                    self.raw.__dict__[k] = loaded[k]
                else:
                    self.raw.__dict__[k] = profile.get(k, self.args[k]["default"])

        for k, v in self.raw.__dict__.items():
            if k not in self.args:
                continue
            c = self.args[k]["choice"]
            if c and not v in c:
                assert False, f"Argument {k}: Invalid value {v}. Allowed: {c}"

        self.parsed = create_recursive_dot_dict({k: self.args[k]["parser"](self.args[k]["type"](v)) if v is not None
                                                 else None for k, v in self.raw.__dict__.items() if k in self.args})

        return self.parsed

    def parse_or_cache(self):
        if self.parsed is None:
            self.do_parse_args()

    def parse(self):
        self.parse_or_cache()
        return self.parsed

    def to_dict(self):
        self.parse_or_cache()
        return self.raw.__dict__

    def clone(self):
        parser = ArgumentParser()
        parser.profiles = self.profiles
        parser.args = self.args
        for name, a in self.args.items():
            parser.parser.add_argument("-" + name, type=int if a["type"] == bool else a["type"], default=None)
        parser.parse()
        return parser

    def from_dict(self, dict):
        return self.do_parse_args(dict)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
            return True

    def load(self, fname):
        if os.path.isfile(fname):
            with open(fname, "r") as data_file:
                map = json.load(data_file)

            self.from_dict(map)
        return self.parsed

    def sync(self, fname=None):
        if fname is None:
            fname = self._get_save_filename()

        if fname is not None:
            if os.path.isfile(fname):
                self.load(fname)

            dir = os.path.dirname(fname)
            os.makedirs(dir, exist_ok=True)

            self.save(fname)
        return self.parsed

    def _get_save_filename(self, opt=None):
        opt = self.parse() if opt is None else opt
        dir = self.get_train_dir(opt)
        return None if dir is None else os.path.join(dir, "args.json")

    def parse_and_sync(self):
        opt = self.parse()
        return self.sync(self._get_save_filename(opt))

    def parse_and_try_load(self):
        fname = self._get_save_filename()
        if fname and os.path.isfile(fname):
            self.load(fname)

        return self.parsed
