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
import cupy
from rmm import _lib as librmm
from rmm._cuda.stream import Stream


# TODO(rongou): move this into RMM.
def rmm_cupy_system_allocator(nbytes):
    """
    A CuPy allocator that makes use of RMM system memory resource.

    Examples
    --------
    >>> from rmm.allocators.cupy import rmm_cupy_allocator
    >>> import cupy
    >>> cupy.cuda.set_allocator(rmm_cupy_system_allocator)
    """
    stream = Stream(obj=cupy.cuda.get_current_stream())
    buf = librmm.device_buffer.DeviceBuffer(size=nbytes, stream=stream)
    mem = cupy.cuda.SystemMemory.from_external(
        ptr=buf.ptr, size=buf.size, owner=buf
    )
    ptr = cupy.cuda.memory.MemoryPointer(mem, 0)

    return ptr
