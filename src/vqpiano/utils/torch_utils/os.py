import subprocess
from typing import Literal


def run_command(command, output_level: Literal["none", "error", "all"] = "error"):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = p.stdout
    stderr = p.stderr
    assert stdout
    assert stderr

    while p.poll() is None:
        if output_level != "none":
            print(stderr.readline().decode("utf-8"), end="")
        if output_level == "all":
            print(stdout.readline().decode("utf-8"), end="")
    if output_level != "none":
        print(stderr.read().decode("utf-8"), end="")
    if output_level == "all":
        print(stdout.read().decode("utf-8"), end="")
    else:
        p.wait()
    assert p.returncode == 0, f"Command failed: {command}"
