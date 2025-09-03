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
