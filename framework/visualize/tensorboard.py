from .. import utils
from ..utils import process
import os
import atexit
from typing import Optional
import shutil

port: Optional[int] = None
tb_process = None


def start(log_dir: str, on_port: Optional[int] = None):
    global port

    global tb_process
    if tb_process is not None:
        return

    port = utils.port.alloc() if on_port is None else on_port

    command = shutil.which("tensorboard")
    if command is None:
        command = os.path.expanduser("~/.local/bin/tensorboard")

    if os.path.isfile(command):
        print("Found tensorboard in", command)
    else:
        assert False, "Tensorboard not found."

    extra_flags = ""
    version = process.run("%s --version" % command, hide_stderr=True, stdout_mode="pipe").communicate()[0].decode()
    if int(version[0])>1:
        extra_flags = "--bind_all"

    print("Starting Tensorboard server on %d" % port)
    tb_process = process.run("%s --port %d --logdir %s %s" % (command, port, log_dir, extra_flags), hide_stderr=True,
                             stdout_mode="hide")
    if not utils.port.wait_for(port):
        print("ERROR: failed to start Tensorboard server. Server not responding.")
        return
    print("Done.")

def kill_tb():
    if tb_process is None:
        return

    tb_process.kill()

atexit.register(kill_tb)
