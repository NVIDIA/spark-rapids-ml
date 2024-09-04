import string
import subprocess
import sys
from pathlib import Path

import spark_rapids_ml


def main_cli() -> None:
    i = 1
    while i < len(sys.argv) and sys.argv[i].startswith("-"):
        if sys.argv[i] in ["--help", "-h", "--version"]:
            output = subprocess.run(
                f"spark-submit {sys.argv[i]}", shell=True, capture_output=True
            ).stderr
            output_str = output.decode("utf-8")
            output_str = output_str.replace("spark-submit", "spark-rapids-submit")
            print(output_str, file=sys.stderr)
            exit(0)
        elif sys.argv[i] in ["--verbose", "-v", "--supervise"]:
            i += 1
        else:
            i += 2

    if i >= len(sys.argv):
        raise ValueError("No application file supplied.")

    command_line = (
        "spark-submit "
        + " ".join(sys.argv[1:i])
        + f" {spark_rapids_ml.__path__[0]}/__main__.py "
        + " ".join(sys.argv[i:])
    )

    print(f"running: {command_line}")

    subprocess.run(command_line, shell=True)
