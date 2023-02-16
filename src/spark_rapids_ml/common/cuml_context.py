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
import asyncio
import base64
import json
import os
import psutil
from typing import Any, Optional

from pylibraft.common import Handle
from pyspark import BarrierTaskContext
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only, inject_comms_on_handle
from raft_dask.common.nccl import nccl
from raft_dask.common import UCX


class CumlContext:
    def __init__(
        self, rank: int, nranks: int, context: BarrierTaskContext, enable: bool, require_ucx: bool = False
    ) -> None:
        """
        Initialize the nccl unique id for workers.

        1. get the nccl unique id for worker 0
        2. do all gather for all the workers to get the nccl unique uid.
        3. if require_ucx is true, initialize ucx and inject ucx together with nccl into a handle
        """
        self.enable = enable
        self._handle: Optional[Handle] = None
        if not enable:
            return 

        self._rank = rank
        self._nranks = nranks
        self._require_ucx = require_ucx

        self._handle = Handle(n_streams=0)
        self._nccl_comm: Optional[nccl] = None
        self._nccl_unique_id = None
        self._ucx = None
        self._ucx_port = None
        self._ucx_eps = None

        nccl_uid = ""
        if context.partitionId() == 0:
            nccl_uid = base64.b64encode(nccl.get_unique_id()).decode("utf-8")

        if self._require_ucx is False:
            nccl_uids = context.allGather(nccl_uid)
            self._nccl_unique_id = base64.b64decode(nccl_uids[0])
        else:
            tasks = context.getTaskInfos()
            ips = [task.address.split(":")[0] for task in tasks]
            my_ip = ips[self._rank]
            my_ifname = CumlContext.get_ifname_from_ip(my_ip)

            # initialize TPC over UCX by setting environmental variables 
            os.environ["UCX_TLS"] = "tcp,cuda_copy,cuda_ipc"
            os.environ["UCXPY_IFNAME"] = my_ifname

            self._ucx = UCX.get()
            self._ucx_port = self._ucx.listener_port()
            msgs = context.allGather(json.dumps((nccl_uid, self._ucx_port)))
            self._nccl_unique_id = base64.b64decode(json.loads(msgs[0])[0])
            ports = [json.loads(msg)[1] for msg in msgs]
            self._ucx_eps = asyncio.run(CumlContext._ucp_create_endpoints(self._ucx, list(zip(ips, ports))))


    @property
    def handle(self) -> Optional[Handle]:
        return self._handle

    def __enter__(self) -> "CumlContext":
        if not self.enable:
            return self

        # initialize nccl and inject it to the handle. A GPU must be assigned exclusively before init() is called
        self._nccl_comm = nccl()
        self._nccl_comm.init(self._nranks, self._nccl_unique_id, self._rank)

        if self._require_ucx is False:
            inject_comms_on_handle_coll_only(
                self._handle, self._nccl_comm, self._nranks, self._rank, True
            )
        else:
            inject_comms_on_handle(
                self._handle, self._nccl_comm, self._ucx.get_worker(), self._ucx_eps, self._nranks, self._rank, True
            )
        return self

    def __exit__(self, *args: Any) -> None:
        if not self.enable:
            return 
        assert self._nccl_comm is not None
        self._nccl_comm.destroy()
        del self._nccl_comm

        del self._handle

    @staticmethod
    def get_ifname_from_ip(target_ip: str):
        if_addrs_dict = psutil.net_if_addrs() 
        for ifname in if_addrs_dict:
            ip = if_addrs_dict[ifname][0].address
            if ip == target_ip:
                return ifname
        raise ValueError("target_ip ${target_ip} does not exist")

    @staticmethod
    async def _ucp_create_endpoints(ucx_worker, target_ip_ports, additional_timeout=0.1):
        """
            ucp initialization may require a larger additional_timeout a complex network environment
        """
        eps = [None] * len(target_ip_ports)
        for i in range(len(eps)):
            ip, port = target_ip_ports[i]
            ep = await ucx_worker.get_endpoint(ip, port)
            eps[i] = ep
        await asyncio.sleep(additional_timeout)
        return eps
