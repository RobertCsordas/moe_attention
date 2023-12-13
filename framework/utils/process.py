import sys
import ctypes
import subprocess
import os


def run(cmd: str, hide_stderr: bool = True, stdout_mode: str = "print"):
    libc_search_dirs = ["/lib", "/lib/x86_64-linux-gnu", "/lib/powerpc64le-linux-gnu"]

    if sys.platform == "linux" :
        found = None
        for d in libc_search_dirs:
            file = os.path.join(d, "libc.so.6")
            if os.path.isfile(file):
                found = file
                break

        if not found:
            print("WARNING: Cannot find libc.so.6. Cannot kill process when parent dies.")
            killer = None
        else:
            libc = ctypes.CDLL(found)
            PR_SET_PDEATHSIG = 1
            KILL = 9
            killer = lambda: libc.prctl(PR_SET_PDEATHSIG, KILL)
    else:
        print("WARNING: OS not linux. Cannot kill process when parent dies.")
        killer = None

    if hide_stderr:
        stderr = open(os.devnull,'w')
    else:
        stderr = None

    if stdout_mode == "hide":
        stdout = open(os.devnull, 'w')
    elif stdout_mode == "print":
        stdout = None
    elif stdout_mode == "pipe":
        stdout = subprocess.PIPE
    else:
        assert False, "Invalid stdout mode: %s" % stdout_mode

    return subprocess.Popen(cmd.split(" "), stderr=stderr, stdout=stdout, preexec_fn=killer)
