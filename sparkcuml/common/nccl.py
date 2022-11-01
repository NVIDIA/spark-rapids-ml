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
import base64
from typing import Any

from pylibraft.common import Handle
from pyspark import BarrierTaskContext
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft_dask.common.nccl import nccl


class NcclComm:
    def __init__(self, nranks: int, context: BarrierTaskContext) -> None:
        """
        Initialize the nccl unique id for workers.

        1. get the nccl unique id for worker 0
        2. do all gather for all the workers to get the nccl unique uid.
        """
        nccl_uid = ""
        if context.partitionId() == 0:
            nccl_uid = base64.b64encode(nccl.get_unique_id()).decode("utf-8")
        nccl_uids = context.allGather(nccl_uid)
        self.nccl_unique_id = base64.b64decode(nccl_uids[0])
        self.nranks = nranks
        self.raft_comm_state: dict[str, Any] = {}

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
            self.raft_comm_state["nccl"] = nccl_comm
        self.raft_comm_state["handle"] = handle
        return handle

    def destroy(self) -> None:
        """
        destroy objects for NCCL communication

        release GPU memory of nccl comm and handle if they exist
        """
        if "nccl" in self.raft_comm_state:
            self.raft_comm_state["nccl"].destroy()
            del self.raft_comm_state["nccl"]

        if "handle" in self.raft_comm_state:
            del self.raft_comm_state["handle"]
