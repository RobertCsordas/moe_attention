from typing import List
from .config import get_config
from .run_command import run_command
import os


def get_ckpt(id: str, ckpt_dir: str):
    config = get_config()

    hostname = config["hostname"]
    wandb_dir = config["wandb_dir"]

    cmd = f"ssh {hostname} 'ls {wandb_dir}/*{id}*/files/checkpoint/model* -t'"
    ckpts = run_command(cmd)

    ckpt = ckpts.splitlines()[0]
    assert dir, "No checkpoint found"

    fname = os.path.basename(ckpt)
    target = f"{ckpt_dir}/{fname}"
    if not os.path.isfile(target):
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Downloading {fname} to {target}")
        run_command(f"scp {hostname}:{ckpt} {target}")

