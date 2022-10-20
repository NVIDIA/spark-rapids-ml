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

from pylibraft.common import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft_dask.common.nccl import nccl


class NcclComm:
    def __init__(self, nranks: int) -> None:
        """
        This class must be instantiated in driver side.

        It will be moved to executor side in the future.
        """
        self.nccl_unique_id = nccl.get_unique_id()
        self.nranks = nranks

    def init_worker(self, rank: int, init_nccl: bool = True) -> Handle:
        """
        Initialize and return a Handle.

        when init_nccl is enabled, it initializes nccl comm, creates a comms_t
        instance and injects it into the handle.
        """

        handle = Handle(n_streams=0)
        if init_nccl:
            nccl_comm = nccl()
            nccl_comm.init(self.nranks, self.nccl_unique_id, rank)
            inject_comms_on_handle_coll_only(handle, nccl_comm, self.nranks, rank, True)

        return handle
