#
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
#

import argparse
import runpy
import sys

import spark_rapids_ml.install


# borrowed from rapids cudf.pandas
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m spark_rapids_ml",
        description=(
            "Run a Python script with Spark RAPIDS ML enabled. "
            "In this mode supported pyspark.ml estimator imports will automatically use GPU acclerated implementations."
        ),
    )

    parser.add_argument(
        "-m",
        dest="module",
        nargs=1,
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass on to the script",
    )

    args = parser.parse_args()

    if args.module:
        (module,) = args.module
        # run the module passing the remaining arguments
        # as if it were run with python -m <module> <args>
        sys.argv[:] = [module] + args.args  # not thread safe?
        runpy.run_module(module, run_name="__main__")
    elif len(args.args) >= 1:
        # Remove ourself from argv and continue
        sys.argv[:] = args.args
        runpy.run_path(args.args[0], run_name="__main__")
    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
