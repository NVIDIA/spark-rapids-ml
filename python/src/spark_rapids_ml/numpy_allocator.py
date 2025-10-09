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

# Configure numpy allocator globally to use cupy allocator for use when SAM is enabled
# see https://docs.cupy.dev/en/stable/user_guide/memory.html#memory-management
# only configure once per process

import ctypes

import cupy._core.numpy_allocator as ac
import numpy_allocator

_lib = ctypes.CDLL(ac.__file__)


class _my_allocator(metaclass=numpy_allocator.type):
    _calloc_ = ctypes.addressof(_lib._calloc)
    _malloc_ = ctypes.addressof(_lib._malloc)
    _realloc_ = ctypes.addressof(_lib._realloc)
    _free_ = ctypes.addressof(_lib._free)


_my_allocator.__enter__()  # change the allocator globally
