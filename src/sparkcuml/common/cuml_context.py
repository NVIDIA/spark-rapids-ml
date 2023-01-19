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
from typing import Any, Optional

from pylibraft.common import Handle
from pyspark import BarrierTaskContext
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft_dask.common.nccl import nccl


class CumlContext:
    def __init__(self, rank: int, nranks: int, context: BarrierTaskContext) -> None:
        """
        Initialize the nccl unique id for workers.

        1. get the nccl unique id for worker 0
        2. do all gather for all the workers to get the nccl unique uid.
        """
        nccl_uid = ""
        if context.partitionId() == 0:
            nccl_uid = base64.b64encode(nccl.get_unique_id()).decode("utf-8")
        nccl_uids = context.allGather(nccl_uid)

        self._nccl_unique_id = base64.b64decode(nccl_uids[0])

        self._rank = rank
        self._nranks = nranks
        self._handle = Handle(n_streams=0)

        self._nccl_comm: Optional[nccl] = None

    @property
    def handle(self) -> Handle:
        return self._handle

    def __enter__(self) -> "CumlContext":
        # initialize nccl and inject it to the handle
        self._nccl_comm = nccl()
        self._nccl_comm.init(self._nranks, self._nccl_unique_id, self._rank)
        inject_comms_on_handle_coll_only(
            self._handle, self._nccl_comm, self._nranks, self._rank, True
        )
        return self

    def __exit__(self, *args: Any) -> None:
        assert self._nccl_comm is not None
        self._nccl_comm.destroy()
        del self._nccl_comm

        del self._handle
