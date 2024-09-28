#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
__version__ = "24.10.0"

import pandas as pd
import pyspark

# patch pandas 2.0+ for backward compatibility with psypark < 3.4
from packaging import version

if version.parse(pyspark.__version__) < version.parse("3.4.0") and version.parse(
    pd.__version__
) >= version.parse("2.0.0"):
    pd.DataFrame.iteritems = pd.DataFrame.items
    pd.Series.iteritems = pd.Series.items
