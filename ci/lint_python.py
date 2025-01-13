# Copyright (c) 2024, NVIDIA CORPORATION.
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

from typing import Dict, List, Tuple

import argparse
from io import StringIO
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pylint.lint import Run
from pylint.reporters.text import TextReporter

# This script is copied from dmlc/xgboost

CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.normpath(os.path.join(CURDIR, os.path.pardir))
SRC_PATHS = [
    "src/spark_rapids_ml",
    "tests",
    "tests_large",
    "benchmark",
    "tests_no_import_change",
]


def run_formatter(rel_paths: List[str]) -> bool:
    isort_cmd = ["isort", "--check", "--profile=black"] + rel_paths
    black_cmd = ["black", "--check"] + rel_paths
    isort_ret = subprocess.run(isort_cmd).returncode
    black_ret = subprocess.run(black_cmd).returncode
    if isort_ret != 0 or black_ret != 0:
        isort_cmd.remove("--check")
        black_cmd.remove("--check")
        msg = (
            "Please run the following command on your machine to address the format"
            " errors:\n {}\n {}".format(" ".join(isort_cmd), " ".join(black_cmd))
        )
        print(msg, file=sys.stdout)
        return False
    return True


def run_mypy(rel_paths: List[str]) -> bool:
    ret = subprocess.run(["mypy"] + rel_paths)
    return ret.returncode == 0


class PyLint:
    """A helper for running pylint, mostly copied from dmlc-core/scripts."""

    def __init__(self) -> None:
        self.pypackage_root = PROJECT_ROOT
        self.pylint_cats = set(["error", "warning", "convention", "refactor"])
        self.pylint_opts = [
            "--extension-pkg-whitelist=numpy",
            "--rcfile=" + os.path.join(self.pypackage_root, ".pylintrc"),
        ]

    def run(self, path: str) -> Tuple[Dict, str, str]:

        pylint_output = StringIO()
        reporter = TextReporter(pylint_output)
        Run([str(path)] + self.pylint_opts, reporter=reporter, exit=False)

        emap = {}
        err = ""
        
        out = []
        for line in pylint_output:
            out.append(line)
            key = line.split(":")[-1].split("(")[0].strip()
            if key not in self.pylint_cats:
                continue
            if key not in emap:
                emap[key] = 1
            else:
                emap[key] += 1

        return {path: emap}, err, "\n".join(out)

    def __call__(self) -> bool:
        all_errors: Dict[str, Dict[str, int]] = {}

        def print_summary_map(result_map: Dict[str, Dict[str, int]]) -> int:
            """Print summary of certain result map."""
            if len(result_map) == 0:
                return 0
            ftype = "Python"
            npass = sum(1 for x in result_map.values() if len(x) == 0)
            print(f"====={npass}/{len(result_map)} {ftype} files passed check=====")
            for fname, emap in result_map.items():
                if len(emap) == 0:
                    continue
                print(
                    f"{fname}: {sum(emap.values())} Errors of {len(emap)} Categories map={str(emap)}"
                )
            return len(result_map) - npass

        all_scripts = []
        for root, dirs, files in os.walk(self.pypackage_root):
            for f in files:
                if f.endswith(".py"):
                    all_scripts.append(os.path.join(root, f))

        with Pool(cpu_count()) as pool:
            error_maps = pool.map(self.run, all_scripts)
            for emap, err, out in error_maps:
                print(out)
                if len(err) != 0:
                    print(err)
                all_errors.update(emap)

        nerr = print_summary_map(all_errors)
        return nerr == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", action="store_true", default=False)
    parser.add_argument("--type-check", action="store_true", default=False)
    parser.add_argument("--pylint", action="store_true", default=False)
    args = parser.parse_args()
    if args.format:
        print("Formatting...")
        if not run_formatter(SRC_PATHS):
            sys.exit(-1)

    if args.type_check:
        print("Type checking...")
        if not run_mypy(SRC_PATHS):
            sys.exit(-1)

    if args.pylint:
        print("Running PyLint...")
        if not PyLint()():
            sys.exit(-1)
