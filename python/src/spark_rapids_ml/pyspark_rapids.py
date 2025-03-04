#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import subprocess
import sys

import spark_rapids_ml


def main_cli() -> None:

    i = 1
    while i < len(sys.argv) and sys.argv[i].startswith("-"):
        if sys.argv[i] in ["--help", "-h", "--version"]:
            output = subprocess.run(
                f"pyspark {sys.argv[i]}", shell=True, capture_output=True
            ).stderr
            output_str = output.decode("utf-8")
            output_str = output_str.replace("pyspark", "pyspark-rapids")
            print(output_str, file=sys.stderr)
            exit(0)
        elif sys.argv[i] in ["--verbose", "-v", "--supervise"]:
            i += 1
        else:
            i += 2

    command_line = "pyspark " + " ".join(sys.argv[1:])
    env = dict(os.environ)
    env["PYTHONSTARTUP"] = f"{spark_rapids_ml.__path__[0]}/install.py"
    subprocess.run(command_line, shell=True, env=env)
