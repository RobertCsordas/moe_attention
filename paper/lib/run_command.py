import subprocess


def run_command(cmd: str, get_stderr: bool = False) -> str:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if get_stderr else None,
                            shell=True, stdin=subprocess.PIPE)
    res = proc.communicate(None)
    stdout = res[0].decode()
    assert proc.returncode == 0, f"Command {cmd} failed with return code {proc.returncode} and stderr"
    return stdout
