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

from raft.dask.common.nccl import nccl
from raft.dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft.common import Handle

class SparkComm:
    def __init__(self):
        self.nccl_unique_id = nccl.get_unique_id()
    
    def init_worker(self, num_workers, wid):
        nccl_comm = nccl()
        nccl_comm.init(num_workers, self.nccl_unique_id, wid)
        handle = Handle(n_streams = 0)
        inject_comms_on_handle_coll_only(handle, nccl_comm, num_workers, wid, True)
        return handle
