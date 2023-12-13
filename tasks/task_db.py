from typing import Optional
import re
import os
import importlib
import inspect
import framework

TASKS = {}
ARGS_REGISTERS = []
TASK_PREFIX = ""


def args(fn):
    global ARGS_REGISTERS
    ARGS_REGISTERS.append(fn)
    return fn


def register_args(parser: framework.helpers.ArgumentParser):
    for f in ARGS_REGISTERS:
        f(parser)


def register_files(prefix=""):
    global TASK_PREFIX

    old_prefix = TASK_PREFIX
    TASK_PREFIX = f"{old_prefix}.{prefix}" if old_prefix else prefix

    caller = inspect.stack()[1]
    caller_dir = os.path.dirname(caller.filename)
    caller_package = inspect.getmodule(inspect.stack()[1][0]).__name__

    for module in os.listdir(caller_dir):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        importlib.import_module("." + module[:-3], package=caller_package)

    TASK_PREFIX = old_prefix


def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def task(name: Optional[str] = None):
    def wrapper(cls):
        n = TASK_PREFIX + (name or camel_to_snake(cls.__name__))
        assert n not in TASKS, f"Task {n} already exists"
        TASKS[n] = cls
        return cls
    return wrapper


def get_task(name: str):
    assert name in TASKS, f"Task {name} doesn't exists"
    return TASKS[name]
